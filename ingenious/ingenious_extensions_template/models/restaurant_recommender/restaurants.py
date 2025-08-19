import json
from typing import Any, List

from pydantic import BaseModel, Field


# A simple utility to convert a list of Pydantic models to a CSV-formatted string.
# This is a stand-in for the `ingenious.utils.model_utils.Listable_Object_To_Csv`
# from your example.
def Listable_Object_To_Csv(data: List[BaseModel], model_class: Any) -> str:
    """Converts a list of Pydantic objects into a CSV string."""
    if not data:
        return ""

    # Get headers from the model's schema
    headers = list(model_class.model_fields.keys())
    csv_lines = [",".join(headers)]

    # Get rows
    for item in data:
        row = []
        for header in headers:
            value = getattr(item, header, "")
            # Handle potential commas in string values by quoting them
            if isinstance(value, str) and "," in value:
                row.append(f'"{value}"')
            else:
                row.append(str(value))
        csv_lines.append(",".join(row))

    return "\n".join(csv_lines)


class RootModel_Budget(BaseModel):
    """Defines the estimated budget for different group sizes."""

    couple: float = Field(..., description="Estimated budget for a couple (2 people)")
    small_group: float = Field(
        ..., description="Estimated budget for a group of 5-10 people"
    )
    large_group: float = Field(
        ..., description="Estimated budget for a group of more than 10 people"
    )


class RootModel_Restaurant(BaseModel):
    """Defines the core details of a single restaurant."""

    name: str
    location: str
    cuisine: str
    budget: RootModel_Budget


class RootModel_Restaurant_Display(BaseModel):
    """A flattened model for displaying restaurant data in a table."""

    name: str
    location: str
    cuisine: str
    budget_for_couple: float
    budget_for_small_group: float
    budget_for_large_group: float


class RootModel(BaseModel):
    """The root model for our restaurant data, containing a list of restaurants."""

    restaurants: List[RootModel_Restaurant]

    @staticmethod
    def load_from_json(json_data: str) -> "RootModel":
        """Loads restaurant data from a JSON string and returns a RootModel instance."""
        data = json.loads(json_data)
        root_model = RootModel(**data)
        return root_model

    def display_restaurants_as_table(self) -> str:
        """Formats the list of restaurants into a markdown table (CSV format)."""
        table_data: List[RootModel_Restaurant_Display] = []

        for restaurant in self.restaurants:
            # Create a flattened record for display purposes
            display_rec = RootModel_Restaurant_Display(
                name=restaurant.name,
                location=restaurant.location,
                cuisine=restaurant.cuisine,
                budget_for_couple=restaurant.budget.couple,
                budget_for_small_group=restaurant.budget.small_group,
                budget_for_large_group=restaurant.budget.large_group,
            )
            table_data.append(display_rec)

        # Convert the list of display models to a CSV string
        csv_output = Listable_Object_To_Csv(table_data, RootModel_Restaurant_Display)

        # Note: Always provide tabular data with a heading for better rendering.
        return "## Restaurant Recommendations\n" + csv_output
