from dash import html, dcc
from typing import List, Optional, Dict


class TeamSelector(html.Div):
    """A component for selecting teams."""

    def __init__(
        self,
        id: str,
        label: str = "Select Team",
        placeholder: str = "Select a team...",
        teams: Optional[List[Dict[str, str]]] = None,
        value: Optional[str] = None,
        className: str = "",
    ):
        """
        Initialize the team selector component.

        Args:
            id: The component ID
            label: The label for the selector
            placeholder: The placeholder text
            teams: List of team dictionaries with 'value' and 'label' keys
            value: The initially selected value
            className: Additional CSS classes
        """
        teams = teams or []

        super().__init__(
            children=[
                html.Label(label, className="form-label"),
                dcc.Dropdown(
                    id=id,
                    options=teams,
                    value=value,
                    placeholder=placeholder,
                    className="dash-dropdown",
                    optionHeight=50,
                    clearable=False,
                ),
            ],
            className=f"mb-3 {className}",
        )


class DateRangeSelector(html.Div):
    """A component for selecting date ranges."""

    def __init__(
        self,
        id: str,
        label: str = "Select Date Range",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        className: str = "",
    ):
        """
        Initialize the date range selector component.

        Args:
            id: The component ID
            label: The label for the selector
            min_date: The minimum selectable date (YYYY-MM-DD)
            max_date: The maximum selectable date (YYYY-MM-DD)
            start_date: The initially selected start date (YYYY-MM-DD)
            end_date: The initially selected end date (YYYY-MM-DD)
            className: Additional CSS classes
        """
        super().__init__(
            children=[
                html.Label(label, className="form-label"),
                dcc.DatePickerRange(
                    id=id,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=start_date,
                    end_date=end_date,
                    className="w-100",
                ),
            ],
            className=f"mb-3 {className}",
        )


class MetricSelector(html.Div):
    """A component for selecting metrics."""

    def __init__(
        self,
        id: str,
        label: str = "Select Metrics",
        metrics: Optional[List[Dict[str, str]]] = None,
        value: Optional[List[str]] = None,
        className: str = "",
    ):
        """
        Initialize the metric selector component.

        Args:
            id: The component ID
            label: The label for the selector
            metrics: List of metric dictionaries with 'value' and 'label' keys
            value: The initially selected values
            className: Additional CSS classes
        """
        metrics = metrics or []

        super().__init__(
            children=[
                html.Label(label, className="form-label"),
                dcc.Dropdown(
                    id=id,
                    options=metrics,
                    value=value,
                    multi=True,
                    className="dash-dropdown",
                    placeholder="Select metrics...",
                ),
            ],
            className=f"mb-3 {className}",
        )
