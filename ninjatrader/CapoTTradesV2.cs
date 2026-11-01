// Capo TTrades v2: the backtester's "nq_ttrades_v2" best setup, for a NinjaTrader 8 SIM / demo account.
// Locked settings (Capo, 2026-11-01): context = 4H sweep OR 4H FVG, mode = reversal + 15M continuation,
// entry on the 15M CISD close, bias = with the 6 PM daily open, stop at the CISD candle, target 3R,
// arm time 8 x 15M bars, Asia + London + NY kill zones, $150 risk (25K), max 2 trades a day (+1 more if the first two both won).
//
// Rules:
//  - Chart: MNQ 15-minute bars, trading hours "CME US Index Futures ETH". Times are converted to New York time.
//  - 4H candles are built from the 15M bars: 6 PM, 10 PM, 2 AM, 6 AM, 10 AM, 2 PM New York.
//  - A side "arms" when a 15M bar trades through the last finished 4H candle's low (long) / high (short), or
//    into the newest unfilled 4H fair value gap. It stays armed for ArmBars 15M bars.
//  - CISD: a 15M CLOSE through the open of the first candle of the latest opposite run of candles
//    (down candles for a long, up candles for a short). Armed + CISD = reversal entry, market at the close.
//  - Continuation: after a reversal CISD, a pullback (counter candle) that holds the reversal extreme, then
//    another CISD in the same direction, within 16 bars.
//  - Bias: longs only when the 15M close is above today's 6 PM open, shorts only below it.
//  - Stop: beyond the CISD candle's low (long) / high (short) + 1 point. Target: TargetR x stop.
//  - Size: floor(RiskPerTrade / (stop x point value)), max MaxContracts; skipped if not even 1 contract.
//  - Flat by 3:55 PM New York. One position at a time.
//  - Session switches (NY time): Asia 8 PM-12 AM, London 2-5 AM, NY window (default 8:30-11). Backtest (3y NQ, $150, Lucid Pro 25K):
//    Asia + London + NY = 69% pass / 22% blow-up; Asia + London = 67% / 22%; London + NY = 60% / 14%; NY alone = 11%.
//    The trading day starts at 6 PM, so Asia trades come first and count toward the daily cap.
//  - Optional (default OFF, still being backtested): move the stop to breakeven once price takes out the most
//    recent 15M swing point (N bars each side) beyond the entry.
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class CapoTTradesV2 : Strategy
    {
        private const double StopPad = 1.0;
        private const int ContBars = 16;

        private TimeZoneInfo ny;
        // 4H candles built from 15M bars
        private readonly List<double> h4o = new List<double>(), h4h = new List<double>(), h4l = new List<double>(), h4c = new List<double>();
        private bool building;
        private double bo, bh, bl, bc;
        private int curBucket = -1;
        private DateTime curDay = DateTime.MinValue;
        private bool hasBull, hasBear;
        private double bullTop, bullBot, bearTop, bearBot;
        private double dayOpen = double.NaN;

        // per side state, 0 = long, 1 = short
        private readonly int[] armed = { -1, -1 };
        private readonly double[] ext = { double.NaN, double.NaN };
        private readonly double[] level = { double.NaN, double.NaN };
        private readonly int[] cstate = { -1, -1 };
        private readonly double[] cext = { double.NaN, double.NaN };
        private readonly double[] pb = { double.NaN, double.NaN };
        private readonly bool[] pbSeen = { false, false };
        private int lastDir;
        private int k = -1; // 15M bar counter
        private int tradesToday, closedToday, winsToday;

        // orders
        private Order entryOrder, stopOrder, targetOrder, flatOrder;
        private double stopPts, tgtPts, fillPx = double.NaN, beLevel = double.NaN;
        private int dir;
        private bool movedToBe;

        // 15M swings for the breakeven option
        private readonly List<double> swingHi = new List<double>(), swingLo = new List<double>();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "CapoTTradesV2";
                Description = "TTrades fractal v2: 4H sweep/FVG + 15M CISD, daily-open bias, CISD-candle stop, 3R, NY only.";
                Calculate = Calculate.OnBarClose;
                IsUnmanaged = true;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 300;
                BarsRequiredToTrade = 40;
                RiskPerTrade = 150;
                MaxContracts = 20;
                TargetR = 3.0;
                ArmBars = 8;
                WindowStart = 830;
                WindowEnd = 1100;
                MaxTrades = 2;
                ExtraTradeAfterTwoWins = true;
                UseNY = true;
                UseLondon = true;
                UseAsia = true;
                AsiaStart = 2000;
                AsiaEnd = 2400;
                LondonStart = 200;
                LondonEnd = 500;
                BreakevenOnSwing = false;
                SwingStrength = 2;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 1);
            }
            else if (State == State.DataLoaded)
            {
                ny = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
            }
        }

        private DateTime ToNy(DateTime t) { return TimeZoneInfo.ConvertTime(t, Core.Globals.GeneralOptions.TimeZoneInfo, ny); }
        private static int Hm(int hhmm) { return (hhmm / 100) * 60 + hhmm % 100; }
        private static bool Active(Order o) { return o != null && !Order.IsTerminalState(o.OrderState); }
        private double Tick(double px) { return Instrument.MasterInstrument.RoundToTickSize(px); }

        private bool CanTrade()
        {
            if (tradesToday < MaxTrades) return true;
            return ExtraTradeAfterTwoWins && MaxTrades == 2 && tradesToday == 2 && closedToday == 2 && winsToday == 2;
        }

        private void FinishCandle()
        {
            if (!building) return;
            h4o.Add(bo); h4h.Add(bh); h4l.Add(bl); h4c.Add(bc);
            building = false;
            int i = h4c.Count - 1;
            if (i >= 2 && h4l[i] > h4h[i - 2]) { hasBull = true; bullTop = h4l[i]; bullBot = h4h[i - 2]; }
            if (i >= 2 && h4h[i] < h4l[i - 2]) { hasBear = true; bearTop = h4l[i - 2]; bearBot = h4h[i]; }
            if (hasBull && h4c[i] < bullBot) hasBull = false;   // filled
            if (hasBear && h4c[i] > bearTop) hasBear = false;
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress == 1)
            {
                if (CurrentBars[1] < 1) return;
                OnMinute();
                return;
            }
            if (CurrentBars[0] < 3) return;
            k++;

            DateTime end = ToNy(Time[0]);
            DateTime start = end.AddMinutes(-15);
            int sm = start.Hour * 60 + start.Minute;
            DateTime tday = sm >= 18 * 60 ? start.Date.AddDays(1) : start.Date;
            int bucket = (((sm - 18 * 60) % 1440) + 1440) % 1440 / 240;

            if (tday != curDay)
            {
                FinishCandle();
                curDay = tday;
                dayOpen = Open[0];
                tradesToday = closedToday = winsToday = 0;
                armed[0] = armed[1] = -1;
                cstate[0] = cstate[1] = -1;
            }
            else if (bucket != curBucket)
                FinishCandle();
            curBucket = bucket;
            if (!building) { building = true; bo = Open[0]; bh = High[0]; bl = Low[0]; bc = Close[0]; }
            else { bh = Math.Max(bh, High[0]); bl = Math.Min(bl, Low[0]); bc = Close[0]; }
            int em = end.Hour * 60 + end.Minute;
            if ((((em - 18 * 60) % 1440) + 1440) % 1440 % 240 == 0)
                FinishCandle();   // this 15M bar closes the 4H candle

            TrackSwings();

            int j = h4c.Count - 1;
            if (j < 0) { TrackRuns(new[] { false, false }); return; }
            double o = Open[0], h = High[0], l = Low[0], c = Close[0];
            double prevHi = h4h[j], prevLo = h4l[j];
            int t = em - 1;   // same clock as the backtester (minute of the last 1M bar in this 15M bar)
            bool inWin = (UseNY && t >= Hm(WindowStart) && t < Hm(WindowEnd)) || (UseLondon && t >= Hm(LondonStart) && t < Hm(LondonEnd))
                         || (UseAsia && t >= Hm(AsiaStart) && t < Hm(AsiaEnd));

            // 1) 4H context arms a side
            bool[] tap = { l < prevLo, h > prevHi };
            if (hasBull && l <= bullTop && h >= bullBot) tap[0] = true;
            if (hasBear && h >= bearBot && l <= bearTop) tap[1] = true;
            for (int s = 0; s < 2; s++)
            {
                if (armed[s] >= 0)
                {
                    ext[s] = s == 0 ? Math.Min(ext[s], l) : Math.Max(ext[s], h);
                    if (!tap[s] && k - armed[s] > ArmBars) armed[s] = -1;
                }
                if (tap[s] && armed[s] < 0) { armed[s] = k; ext[s] = s == 0 ? l : h; }
            }

            // 2) CISD + bias
            bool[] cisd = { !double.IsNaN(level[0]) && c > level[0], !double.IsNaN(level[1]) && c < level[1] };
            bool[] fired = { false, false };
            bool[] allow = { c > dayOpen, c < dayOpen };

            for (int s = 0; s < 2; s++)
            {
                if (cstate[s] >= 0 && k > cstate[s])
                {
                    bool broke = s == 0 ? l < cext[s] : h > cext[s];
                    if (broke || k - cstate[s] > ContBars) cstate[s] = -1;
                    else if (cisd[s] && pbSeen[s])
                    {
                        fired[s] = true;
                        double sp = (s == 0 ? c - l : h - c) + StopPad;
                        if (inWin && CanTrade() && allow[s]) { Place(s, sp); tradesToday++; }
                        cstate[s] = -1;
                    }
                    else
                    {
                        if (s == 0 ? c < o : c > o) pbSeen[s] = true;
                        if (pbSeen[s]) pb[s] = s == 0 ? Math.Min(pb[s], l) : Math.Max(pb[s], h);
                    }
                }
                if (armed[s] >= 0 && cisd[s])
                {
                    fired[s] = true;
                    double sp = (s == 0 ? c - l : h - c) + StopPad;
                    if (inWin && CanTrade() && allow[s]) { Place(s, sp); tradesToday++; }
                    armed[s] = -1;
                    cstate[s] = k; cext[s] = ext[s]; pbSeen[s] = false;
                    pb[s] = s == 0 ? double.PositiveInfinity : double.NegativeInfinity;
                }
            }
            TrackRuns(fired);
        }

        private void TrackRuns(bool[] fired)
        {
            for (int s = 0; s < 2; s++) if (fired[s]) level[s] = double.NaN;
            int cur = Close[0] > Open[0] ? 1 : (Close[0] < Open[0] ? -1 : 0);
            if (cur == -1 && lastDir != -1) level[0] = Open[0];
            if (cur == 1 && lastDir != 1) level[1] = Open[0];
            if (cur != 0) lastDir = cur;
        }

        private void TrackSwings()
        {
            int n = SwingStrength;
            if (CurrentBars[0] < 2 * n) return;
            bool isHi = true, isLo = true;
            for (int q = 0; q <= 2 * n; q++)
            {
                if (q == n) continue;
                if (High[n] <= High[q]) isHi = false;
                if (Low[n] >= Low[q]) isLo = false;
            }
            if (isHi) { swingHi.Add(High[n]); if (swingHi.Count > 50) swingHi.RemoveAt(0); }
            if (isLo) { swingLo.Add(Low[n]); if (swingLo.Count > 50) swingLo.RemoveAt(0); }
        }

        private void Place(int s, double sp)
        {
            // one position at a time, like the backtest: a signal while in a trade still counts as a trade of the day
            if (sp <= 0 || Position.MarketPosition != MarketPosition.Flat || Active(entryOrder)) return;
            int qty = (int)Math.Floor(RiskPerTrade / (sp * Instrument.MasterInstrument.PointValue));
            qty = Math.Min(qty, MaxContracts);
            if (qty < 1) return;   // stop too wide for even 1 micro: skip, like the backtest
            stopPts = sp;
            tgtPts = TargetR * sp;
            dir = s == 0 ? 1 : -1;
            entryOrder = SubmitOrderUnmanaged(1, s == 0 ? OrderAction.Buy : OrderAction.SellShort, OrderType.Market, qty, 0, 0, "",
                                              s == 0 ? "TT long" : "TT short");
        }

        private void OnMinute()
        {
            int m = ToNy(Times[1][0]).Hour * 60 + ToNy(Times[1][0]).Minute;
            bool flatTime = m >= 15 * 60 + 55 && m < 18 * 60;
            if (Position.MarketPosition != MarketPosition.Flat && !Active(flatOrder))
            {
                if (flatTime)
                {
                    if (Active(stopOrder)) CancelOrder(stopOrder);
                    if (Active(targetOrder)) CancelOrder(targetOrder);
                    flatOrder = SubmitOrderUnmanaged(1, Position.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover,
                                                     OrderType.Market, Position.Quantity, 0, 0, "", "Flat 3:55");
                    return;
                }
                if (BreakevenOnSwing && !movedToBe && !double.IsNaN(beLevel) && Active(stopOrder))
                {
                    bool taken = dir == 1 ? Highs[1][0] > beLevel : Lows[1][0] < beLevel;
                    if (taken)
                    {
                        ChangeOrder(stopOrder, stopOrder.Quantity, 0, Tick(fillPx));
                        movedToBe = true;
                    }
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
                                                  MarketPosition marketPosition, string orderId, DateTime time)
        {
            Order o = execution.Order;
            if (o == entryOrder && o.OrderState == OrderState.Filled)
            {
                fillPx = o.AverageFillPrice;
                string oco = "X" + Guid.NewGuid().ToString("N");
                double sl = Tick(fillPx - dir * stopPts);
                double tp = Tick(fillPx + dir * tgtPts);
                OrderAction exit = dir == 1 ? OrderAction.Sell : OrderAction.BuyToCover;
                stopOrder = SubmitOrderUnmanaged(1, exit, OrderType.StopMarket, o.Filled, 0, sl, oco, "Stop loss");
                targetOrder = SubmitOrderUnmanaged(1, exit, OrderType.Limit, o.Filled, tp, 0, oco, "Take profit");
                movedToBe = false;
                beLevel = double.NaN;
                List<double> sw = dir == 1 ? swingHi : swingLo;   // most recent 15M swing beyond the entry
                for (int q = sw.Count - 1; q >= 0; q--)
                    if (dir == 1 ? sw[q] > fillPx : sw[q] < fillPx) { beLevel = sw[q]; break; }
                entryOrder = null;
            }
            if (Position.MarketPosition == MarketPosition.Flat && (o == stopOrder || o == targetOrder || o == flatOrder)
                && o.OrderState == OrderState.Filled)
            {
                double pnl = dir * (o.AverageFillPrice - fillPx);
                closedToday++;
                if (pnl > 0) winsToday++;
                if (Active(stopOrder)) CancelOrder(stopOrder);
                if (Active(targetOrder)) CancelOrder(targetOrder);
                stopOrder = targetOrder = flatOrder = null;
            }
        }

        #region Settings
        [NinjaScriptProperty, Range(1, double.MaxValue), Display(Name = "Risk per trade ($)", Order = 1, GroupName = "Settings")]
        public double RiskPerTrade { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "Max contracts (micros)", Order = 2, GroupName = "Settings")]
        public int MaxContracts { get; set; }

        [NinjaScriptProperty, Range(0.5, double.MaxValue), Display(Name = "Take profit (x stop)", Order = 3, GroupName = "Settings")]
        public double TargetR { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "4H tap stays armed (15M bars)", Order = 4, GroupName = "Settings")]
        public int ArmBars { get; set; }

        [NinjaScriptProperty, Range(0, 2359), Display(Name = "Trade from (NY, hhmm)", Order = 5, GroupName = "Settings")]
        public int WindowStart { get; set; }

        [NinjaScriptProperty, Range(0, 2400), Display(Name = "Trade until (NY, hhmm)", Order = 6, GroupName = "Settings")]
        public int WindowEnd { get; set; }

        [NinjaScriptProperty, Range(1, 10), Display(Name = "Max trades a day", Order = 7, GroupName = "Settings")]
        public int MaxTrades { get; set; }

        [NinjaScriptProperty, Display(Name = "Allow a 3rd trade after 2 wins", Order = 8, GroupName = "Settings")]
        public bool ExtraTradeAfterTwoWins { get; set; }

        [NinjaScriptProperty, Display(Name = "Breakeven after 15M swing taken (testing)", Order = 9, GroupName = "Settings")]
        public bool BreakevenOnSwing { get; set; }

        [NinjaScriptProperty, Range(1, 10), Display(Name = "Swing strength (bars each side)", Order = 10, GroupName = "Settings")]
        public int SwingStrength { get; set; }

        [NinjaScriptProperty, Display(Name = "Trade London kill zone", Order = 11, GroupName = "Settings")]
        public bool UseLondon { get; set; }

        [NinjaScriptProperty, Range(0, 2359), Display(Name = "London from (NY, hhmm)", Order = 12, GroupName = "Settings")]
        public int LondonStart { get; set; }

        [NinjaScriptProperty, Range(0, 2400), Display(Name = "London until (NY, hhmm)", Order = 13, GroupName = "Settings")]
        public int LondonEnd { get; set; }

        [NinjaScriptProperty, Display(Name = "Trade NY window", Order = 14, GroupName = "Settings")]
        public bool UseNY { get; set; }

        [NinjaScriptProperty, Display(Name = "Trade Asia session", Order = 15, GroupName = "Settings")]
        public bool UseAsia { get; set; }

        [NinjaScriptProperty, Range(0, 2359), Display(Name = "Asia from (NY, hhmm)", Order = 16, GroupName = "Settings")]
        public int AsiaStart { get; set; }

        [NinjaScriptProperty, Range(0, 2400), Display(Name = "Asia until (NY, hhmm)", Order = 17, GroupName = "Settings")]
        public int AsiaEnd { get; set; }
        #endregion
    }
}
