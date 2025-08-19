import json
from datetime import date
from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field


# A simple utility to convert a list of Pydantic models to a CSV-formatted string.
# This is a stand-in for the `ingenious.utils.model_utils.Listable_Object_To_Csv`
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


class RootModel_TaskStatus(str, Enum):
    """Enumeration for the status of a project task."""

    TO_DO = "To Do"
    IN_PROGRESS = "In Progress"
    IN_REVIEW = "In Review"
    DONE = "Done"


class RootModel_TeamMember(BaseModel):
    """Defines a single team member working on a project."""

    name: str = Field(..., description="Full name of the team member.")
    role: str = Field(
        ...,
        description="The role or title of the team member (e.g., 'Developer', 'Designer').",
    )


class RootModel_Task(BaseModel):
    """Defines a single task within a project."""

    id: int = Field(..., description="Unique identifier for the task.")
    title: str = Field(..., description="A brief, descriptive title for the task.")
    assignee: str = Field(
        ..., description="Name of the team member assigned to the task."
    )
    status: RootModel_TaskStatus = Field(
        ..., description="The current status of the task."
    )
    due_date: date = Field(..., description="The target completion date for the task.")


class RootModel_Milestone(BaseModel):
    """Defines a significant milestone in the project timeline."""

    name: str = Field(
        ...,
        description="Name of the milestone (e.g., 'Alpha Release', 'User Testing').",
    )
    target_date: date = Field(
        ..., description="The date this milestone is expected to be reached."
    )
    description: str = Field(
        ..., description="A brief description of what this milestone signifies."
    )


class RootModel_Project(BaseModel):
    """Defines the core details of a single project."""

    project_name: str = Field(..., description="The official name of the project.")
    project_manager: str = Field(
        ..., description="The name of the person leading the project."
    )
    start_date: date = Field(..., description="The start date of the project.")
    end_date: date = Field(..., description="The projected end date of the project.")
    budget: float = Field(
        ..., description="The total allocated budget for the project in USD."
    )
    team: List[RootModel_TeamMember] = Field(
        ..., description="A list of team members on the project."
    )
    tasks: List[RootModel_Task] = Field(
        ..., description="A list of tasks to be completed for the project."
    )
    milestones: List[RootModel_Milestone] = Field(
        ..., description="A list of key project milestones."
    )


class RootModel_Project_Display(BaseModel):
    """A flattened model for displaying a summary of project data in a table."""

    project_name: str
    project_manager: str
    start_date: date
    end_date: date
    budget: float
    team_size: int
    task_count: int
    milestone_count: int


class RootModel(BaseModel):
    """The root model for our project management data, containing a list of projects."""

    projects: List[RootModel_Project]

    @staticmethod
    def load_from_json(json_data: str) -> "RootModel":
        """Loads project data from a JSON string and returns a RootModel instance."""
        data = json.loads(json_data)
        root_model = RootModel(**data)
        return root_model

    def display_projects_as_table(self) -> str:
        """Formats the list of projects into a markdown table (CSV format)."""
        table_data: List[RootModel_Project_Display] = []

        for project in self.projects:
            # Create a flattened summary record for display purposes
            display_rec = RootModel_Project_Display(
                project_name=project.project_name,
                project_manager=project.project_manager,
                start_date=project.start_date,
                end_date=project.end_date,
                budget=project.budget,
                team_size=len(project.team),
                task_count=len(project.tasks),
                milestone_count=len(project.milestones),
            )
            table_data.append(display_rec)

        # Convert the list of display models to a CSV string
        csv_output = Listable_Object_To_Csv(table_data, RootModel_Project_Display)

        return "## Project Portfolio Overview\n" + csv_output
