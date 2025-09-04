from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Submission(BaseModel):
    """Individual submission to be evaluated"""

    submission_id: str = Field(..., description="Unique identifier for the submission")
    title: str | None = Field(
        default=None, description="Optional title or name of the submission"
    )
    content: str = Field(..., description="Main content/description of the submission")
    author: str | None = Field(
        default=None, description="Optional author of the submission"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating submissions"""

    criteria_id: str = Field(..., description="Unique identifier for the criteria")
    name: str = Field(..., description="Name of the criteria")
    description: str = Field(
        ..., description="Detailed description of what to evaluate"
    )
    weight: float = Field(
        default=1.0, description="Weight of this criteria in final scoring (0-1)"
    )
    scoring_guidelines: str = Field(
        ..., description="Guidelines for scoring this criteria"
    )


class SubmissionScore(BaseModel):
    """Score for a specific submission against a criteria"""

    submission_id: str
    criteria_id: str
    score: float = Field(..., ge=0, le=10, description="Score from 0-10")
    reasoning: str = Field(..., description="Explanation for the score")
    feedback: str = Field(default="", description="Additional feedback")


class SubmissionEvaluation(BaseModel):
    """Complete evaluation of a submission"""

    submission_id: str
    scores: List[SubmissionScore]
    total_score: float = Field(default=0.0, description="Weighted total score")
    overall_feedback: str = Field(default="", description="Overall feedback summary")
    rank: int = Field(default=0, description="Rank among all submissions")


class SubmissionOverCriteriaRequest(BaseModel):
    """Main request model for submission evaluation workflow"""

    revision_id: str = Field(..., description="Revision ID for template versioning")
    identifier: str = Field(
        ..., description="Unique identifier for this evaluation session"
    )
    submissions: List[Submission] = Field(
        ..., min_items=1, description="List of submissions to evaluate"
    )
    criteria: List[EvaluationCriteria] = Field(
        ..., min_items=1, description="List of evaluation criteria"
    )
    evaluation_instructions: str = Field(
        default="Evaluate each submission fairly and provide constructive feedback",
        description="Additional instructions for the evaluation process",
    )

    def display_submissions_as_table(self) -> str:
        """Display submissions in a table format for agents"""
        if not self.submissions:
            return "## Submissions\nNo submissions provided."

        # Determine which columns to include based on available data
        has_title = any(submission.title for submission in self.submissions)
        has_author = any(submission.author for submission in self.submissions)

        # Build headers dynamically
        headers = ["ID"]
        if has_title:
            headers.append("Title")
        if has_author:
            headers.append("Author")
        headers.append("Content Preview")

        table_data = []
        for submission in self.submissions:
            row_data = {"ID": submission.submission_id}

            if has_title:
                row_data["Title"] = submission.title if submission.title else ""
            if has_author:
                row_data["Author"] = submission.author if submission.author else ""

            row_data["Content Preview"] = (
                submission.content[:100] + "..."
                if len(submission.content) > 100
                else submission.content
            )
            table_data.append(row_data)

        # Create markdown table
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = []
        for item in table_data:
            row_data = [str(item.get(header, "")) for header in headers]
            rows.append("| " + " | ".join(row_data) + " |")

        table_content = "\n".join([header_row, separator_row] + rows)
        return "## Submissions\n" + table_content

    def display_criteria_as_table(self) -> str:
        """Display evaluation criteria in a table format for agents"""
        table_data = []
        for criteria in self.criteria:
            table_data.append(
                {
                    "ID": criteria.criteria_id,
                    "Name": criteria.name,
                    "Weight": f"{criteria.weight:.2f}",
                    "Description": criteria.description[:100] + "..."
                    if len(criteria.description) > 100
                    else criteria.description,
                }
            )

        # Create markdown table
        if not table_data:
            return "## Evaluation Criteria\nNo criteria provided."

        headers = list(table_data[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = []
        for item in table_data:
            row_data = [str(item.get(header, "")) for header in headers]
            rows.append("| " + " | ".join(row_data) + " |")

        table_content = "\n".join([header_row, separator_row] + rows)
        return "## Evaluation Criteria\n" + table_content


class SubmissionEvaluationResults(BaseModel):
    """Final results of the submission evaluation"""

    evaluations: List[SubmissionEvaluation]
    winning_submission: Submission
    evaluation_summary: str = Field(
        default="", description="Summary of the evaluation process"
    )

    def display_results_as_table(self) -> str:
        """Display final results in a table format"""
        table_data = []
        for evaluation in sorted(self.evaluations, key=lambda x: x.rank):
            table_data.append(
                {
                    "Rank": str(evaluation.rank),
                    "Submission ID": evaluation.submission_id,
                    "Total Score": f"{evaluation.total_score:.2f}",
                    "Overall Feedback": evaluation.overall_feedback[:100] + "..."
                    if len(evaluation.overall_feedback) > 100
                    else evaluation.overall_feedback,
                }
            )

        # Create markdown table
        if not table_data:
            return "## Evaluation Results\nNo results available."

        headers = list(table_data[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = []
        for item in table_data:
            row_data = [str(item.get(header, "")) for header in headers]
            rows.append("| " + " | ".join(row_data) + " |")

        table_content = "\n".join([header_row, separator_row] + rows)
        return "## Evaluation Results\n" + table_content
