"""
DCA Scheduler

Calculates next execution times based on frequency configuration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from .models import DCAConfig, DCAFrequency


class DCAScheduler:
    """Calculates DCA execution schedules."""

    @staticmethod
    def get_next_execution(
        config: DCAConfig,
        after: Optional[datetime] = None,
    ) -> datetime:
        """
        Calculate the next execution time based on frequency.

        Args:
            config: DCA configuration with frequency settings
            after: Calculate next execution after this time (default: now)

        Returns:
            Next scheduled execution datetime (UTC)
        """
        if after is None:
            after = datetime.utcnow()

        frequency = config.frequency

        if frequency == DCAFrequency.HOURLY:
            return DCAScheduler._next_hourly(after)

        elif frequency == DCAFrequency.DAILY:
            return DCAScheduler._next_daily(after, config.execution_hour_utc)

        elif frequency == DCAFrequency.WEEKLY:
            return DCAScheduler._next_weekly(
                after,
                config.execution_hour_utc,
                config.execution_day_of_week or 0,  # Default Sunday
            )

        elif frequency == DCAFrequency.BIWEEKLY:
            return DCAScheduler._next_biweekly(
                after,
                config.execution_hour_utc,
                config.execution_day_of_week or 0,
            )

        elif frequency == DCAFrequency.MONTHLY:
            return DCAScheduler._next_monthly(
                after,
                config.execution_hour_utc,
                config.execution_day_of_month or 1,  # Default 1st
            )

        elif frequency == DCAFrequency.CUSTOM:
            if config.cron_expression:
                return DCAScheduler._next_from_cron(after, config.cron_expression)
            else:
                # Fall back to daily if no cron
                return DCAScheduler._next_daily(after, config.execution_hour_utc)

        # Default fallback
        return DCAScheduler._next_daily(after, config.execution_hour_utc)

    @staticmethod
    def _next_hourly(after: datetime) -> datetime:
        """Next execution at the start of the next hour."""
        next_hour = after.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour

    @staticmethod
    def _next_daily(after: datetime, hour_utc: int) -> datetime:
        """Next execution at specified hour UTC."""
        today_at_hour = after.replace(hour=hour_utc, minute=0, second=0, microsecond=0)

        if after < today_at_hour:
            return today_at_hour
        else:
            return today_at_hour + timedelta(days=1)

    @staticmethod
    def _next_weekly(after: datetime, hour_utc: int, day_of_week: int) -> datetime:
        """Next execution on specified day of week at hour.

        Args:
            day_of_week: 0=Monday, 1=Tuesday, ..., 6=Sunday
        """
        # Python weekday: 0=Monday, 6=Sunday
        current_weekday = after.weekday()

        # Days until target day
        days_until = (day_of_week - current_weekday) % 7

        target_date = after.date() + timedelta(days=days_until)
        target_dt = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour_utc,
            0,
            0,
        )

        # If target is today but time has passed, go to next week
        if target_dt <= after:
            target_dt += timedelta(weeks=1)

        return target_dt

    @staticmethod
    def _next_biweekly(after: datetime, hour_utc: int, day_of_week: int) -> datetime:
        """Next execution every two weeks on specified day."""
        # Get next weekly first
        next_week = DCAScheduler._next_weekly(after, hour_utc, day_of_week)

        # For biweekly, we need to track which week we're on
        # Simple approach: use week number parity
        # This ensures consistent biweekly schedule
        week_number = next_week.isocalendar()[1]

        if week_number % 2 == 0:
            return next_week
        else:
            return next_week + timedelta(weeks=1)

    @staticmethod
    def _next_monthly(after: datetime, hour_utc: int, day_of_month: int) -> datetime:
        """Next execution on specified day of month."""
        # Clamp day to valid range (handle months with fewer days)
        import calendar

        year = after.year
        month = after.month

        # Get max days in current month
        max_day = calendar.monthrange(year, month)[1]
        target_day = min(day_of_month, max_day)

        target_dt = datetime(year, month, target_day, hour_utc, 0, 0)

        if target_dt <= after:
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

            max_day = calendar.monthrange(year, month)[1]
            target_day = min(day_of_month, max_day)
            target_dt = datetime(year, month, target_day, hour_utc, 0, 0)

        return target_dt

    @staticmethod
    def _next_from_cron(after: datetime, cron_expression: str) -> datetime:
        """Calculate next execution from cron expression.

        Simple cron parser supporting: minute hour day month weekday
        """
        try:
            from croniter import croniter

            cron = croniter(cron_expression, after)
            return cron.get_next(datetime)
        except ImportError:
            # Fallback if croniter not installed
            # Default to next hour
            return DCAScheduler._next_hourly(after)
        except Exception:
            # Invalid cron, default to next hour
            return DCAScheduler._next_hourly(after)

    @staticmethod
    def get_estimated_executions_per_year(config: DCAConfig) -> int:
        """Estimate number of executions per year for budget planning."""
        frequency = config.frequency

        if frequency == DCAFrequency.HOURLY:
            return 365 * 24  # 8760
        elif frequency == DCAFrequency.DAILY:
            return 365
        elif frequency == DCAFrequency.WEEKLY:
            return 52
        elif frequency == DCAFrequency.BIWEEKLY:
            return 26
        elif frequency == DCAFrequency.MONTHLY:
            return 12
        elif frequency == DCAFrequency.CUSTOM:
            # Assume daily for custom
            return 365
        else:
            return 52  # Default to weekly

    @staticmethod
    def validate_schedule(config: DCAConfig) -> Optional[str]:
        """Validate schedule configuration.

        Returns error message if invalid, None if valid.
        """
        if config.execution_hour_utc < 0 or config.execution_hour_utc > 23:
            return f"Invalid execution hour: {config.execution_hour_utc}. Must be 0-23."

        if config.frequency == DCAFrequency.WEEKLY:
            if config.execution_day_of_week is not None:
                if config.execution_day_of_week < 0 or config.execution_day_of_week > 6:
                    return f"Invalid day of week: {config.execution_day_of_week}. Must be 0-6."

        if config.frequency == DCAFrequency.MONTHLY:
            if config.execution_day_of_month is not None:
                if config.execution_day_of_month < 1 or config.execution_day_of_month > 31:
                    return f"Invalid day of month: {config.execution_day_of_month}. Must be 1-31."

        if config.frequency == DCAFrequency.CUSTOM:
            if not config.cron_expression:
                return "Custom frequency requires a cron expression."

            try:
                from croniter import croniter

                croniter(config.cron_expression)
            except ImportError:
                pass  # Can't validate without croniter
            except Exception as e:
                return f"Invalid cron expression: {e}"

        return None

    @staticmethod
    def format_schedule_description(config: DCAConfig) -> str:
        """Generate human-readable schedule description."""
        freq = config.frequency
        hour = config.execution_hour_utc

        hour_str = f"{hour:02d}:00 UTC"

        if freq == DCAFrequency.HOURLY:
            return "Every hour"

        elif freq == DCAFrequency.DAILY:
            return f"Daily at {hour_str}"

        elif freq == DCAFrequency.WEEKLY:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day = days[config.execution_day_of_week or 0]
            return f"Weekly on {day} at {hour_str}"

        elif freq == DCAFrequency.BIWEEKLY:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day = days[config.execution_day_of_week or 0]
            return f"Every two weeks on {day} at {hour_str}"

        elif freq == DCAFrequency.MONTHLY:
            day = config.execution_day_of_month or 1
            suffix = "th"
            if day == 1 or day == 21 or day == 31:
                suffix = "st"
            elif day == 2 or day == 22:
                suffix = "nd"
            elif day == 3 or day == 23:
                suffix = "rd"
            return f"Monthly on the {day}{suffix} at {hour_str}"

        elif freq == DCAFrequency.CUSTOM:
            return f"Custom: {config.cron_expression}"

        return "Unknown schedule"
