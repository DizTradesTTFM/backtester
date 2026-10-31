// Capo NQ OCO Open: the backtester's best "slow but safe" setup, for a NinjaTrader 8 SIM account.
// Backtester name: qj_o14_tuned_oco_dip_breakout  (tf 30, NY open, longs only, stop 1.5 ATR, 3R, hold 2x)
//
// Rules (same as the backtest):
//  - Chart: MNQ 30-minute bars, trading hours "CME US Index Futures ETH". Times are converted to New York time.
//  - Every 5th 30-min bar, if the bar closes between 9:30 and 11:29 New York and you are flat with no orders working:
//      place a BUY LIMIT 1.0 ATR below the close and a BUY STOP 1.0 ATR above it (one cancels the other).
//      Unfilled orders are cancelled after 150 minutes.
//  - Stop loss 1.5 x ATR(14) below the fill, take profit 3 x the stop distance above the fill.
//  - Exit after 600 minutes in the trade, and always flat by 3:55 PM New York.
//  - Size: contracts = floor(RiskPerTrade / (stop distance x point value)), max MaxContracts. $300 on a 50K.
// Note: which bar counts as "every 5th" depends on how much history the chart loads, exactly like the backtest
// it is an arbitrary phase. It changes which days trade, not the edge.
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class CapoNqOcoOpen : Strategy
    {
        private ATR atr;
        private Order entryLimit, entryStop, stopOrder, targetOrder, flatOrder;
        private double stopPts, tgtPts;
        private int orderBar = -1;   // 1-minute bar count when the OCO pair was placed
        private int entryBar = -1;   // 1-minute bar count when the entry filled
        private TimeZoneInfo ny;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "CapoNqOcoOpen";
                Description = "NQ 30-min OCO dip-or-breakout, NY open only, longs, 1.5 ATR stop, 3R. Backtester best safe setup.";
                Calculate = Calculate.OnBarClose;
                IsUnmanaged = true;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 300;
                BarsRequiredToTrade = 20;
                RiskPerTrade = 300;
                MaxContracts = 40;
                StopAtr = 1.5;
                TargetR = 3.0;
                OrderDistanceAtr = 1.0;
                EveryNBars = 5;
                ExpiryMinutes = 150;
                MaxHoldMinutes = 600;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 1);
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(BarsArray[0], 14);
                ny = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
            }
        }

        private int NyMinute(DateTime t)
        {
            DateTime n = TimeZoneInfo.ConvertTime(t, Core.Globals.GeneralOptions.TimeZoneInfo, ny);
            return n.Hour * 60 + n.Minute;
        }

        private static bool Active(Order o) { return o != null && !Order.IsTerminalState(o.OrderState); }

        protected override void OnBarUpdate()
        {
            if (CurrentBars[0] < BarsRequiredToTrade || CurrentBars[1] < 1)
                return;

            if (BarsInProgress == 1)
            {
                int m = NyMinute(Times[1][0]);
                bool flatTime = m >= 15 * 60 + 55 && m < 18 * 60;
                if (Active(entryLimit) || Active(entryStop))
                {
                    if (flatTime || CurrentBars[1] - orderBar >= ExpiryMinutes)
                    {
                        if (Active(entryLimit)) CancelOrder(entryLimit);
                        if (Active(entryStop)) CancelOrder(entryStop);
                    }
                }
                if (Position.MarketPosition == MarketPosition.Long && !Active(flatOrder))
                {
                    if (flatTime || (entryBar >= 0 && CurrentBars[1] - entryBar >= MaxHoldMinutes))
                    {
                        if (Active(stopOrder)) CancelOrder(stopOrder);
                        if (Active(targetOrder)) CancelOrder(targetOrder);
                        flatOrder = SubmitOrderUnmanaged(1, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "Time exit");
                    }
                }
                return;
            }

            // 30-minute bar just closed
            if (CurrentBars[0] % EveryNBars != EveryNBars - 1)
                return;
            int close = NyMinute(Times[0][0]);
            if (close < 9 * 60 + 30 || close >= 11 * 60 + 30)
                return;
            if (Position.MarketPosition != MarketPosition.Flat || Active(entryLimit) || Active(entryStop))
                return;

            double a = atr[0];
            double u = a * 0.5;
            stopPts = StopAtr * a;
            tgtPts = TargetR * stopPts;
            int qty = (int)Math.Floor(RiskPerTrade / (stopPts * Instrument.MasterInstrument.PointValue));
            qty = Math.Min(qty, MaxContracts);
            if (qty < 1 || stopPts <= 0)
                return;   // stop too wide for even one contract: skip, like the backtest

            double d = 2.0 * u * OrderDistanceAtr;
            double lim = Instrument.MasterInstrument.RoundToTickSize(Closes[0][0] - d);
            double stp = Instrument.MasterInstrument.RoundToTickSize(Closes[0][0] + d);
            string oco = "E" + Guid.NewGuid().ToString("N");
            entryLimit = SubmitOrderUnmanaged(1, OrderAction.Buy, OrderType.Limit, qty, lim, 0, oco, "Dip limit");
            entryStop = SubmitOrderUnmanaged(1, OrderAction.Buy, OrderType.StopMarket, qty, 0, stp, oco, "Breakout stop");
            orderBar = CurrentBars[1];
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
                                                  MarketPosition marketPosition, string orderId, DateTime time)
        {
            Order o = execution.Order;
            if ((o == entryLimit || o == entryStop) && o.OrderState == OrderState.Filled)
            {
                double fill = o.AverageFillPrice;
                string oco = "X" + Guid.NewGuid().ToString("N");
                double sl = Instrument.MasterInstrument.RoundToTickSize(fill - stopPts);
                double tp = Instrument.MasterInstrument.RoundToTickSize(fill + tgtPts);
                stopOrder = SubmitOrderUnmanaged(1, OrderAction.Sell, OrderType.StopMarket, o.Filled, 0, sl, oco, "Stop loss");
                targetOrder = SubmitOrderUnmanaged(1, OrderAction.Sell, OrderType.Limit, o.Filled, tp, 0, oco, "Take profit");
                entryBar = CurrentBars[1];
                entryLimit = entryStop = null;
            }
            if (Position.MarketPosition == MarketPosition.Flat && (o == stopOrder || o == targetOrder || o == flatOrder))
            {
                if (Active(stopOrder)) CancelOrder(stopOrder);
                if (Active(targetOrder)) CancelOrder(targetOrder);
                stopOrder = targetOrder = flatOrder = null;
                entryBar = -1;
            }
        }

        #region Settings
        [NinjaScriptProperty, Range(1, double.MaxValue), Display(Name = "Risk per trade ($)", Order = 1, GroupName = "Settings")]
        public double RiskPerTrade { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "Max contracts (micros)", Order = 2, GroupName = "Settings")]
        public int MaxContracts { get; set; }

        [NinjaScriptProperty, Range(0.1, double.MaxValue), Display(Name = "Stop (x ATR)", Order = 3, GroupName = "Settings")]
        public double StopAtr { get; set; }

        [NinjaScriptProperty, Range(0.1, double.MaxValue), Display(Name = "Take profit (x stop)", Order = 4, GroupName = "Settings")]
        public double TargetR { get; set; }

        [NinjaScriptProperty, Range(0.1, double.MaxValue), Display(Name = "Order distance (x ATR)", Order = 5, GroupName = "Settings")]
        public double OrderDistanceAtr { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "Place orders every N bars", Order = 6, GroupName = "Settings")]
        public int EveryNBars { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "Cancel unfilled orders after (min)", Order = 7, GroupName = "Settings")]
        public int ExpiryMinutes { get; set; }

        [NinjaScriptProperty, Range(1, int.MaxValue), Display(Name = "Max time in trade (min)", Order = 8, GroupName = "Settings")]
        public int MaxHoldMinutes { get; set; }
        #endregion
    }
}
