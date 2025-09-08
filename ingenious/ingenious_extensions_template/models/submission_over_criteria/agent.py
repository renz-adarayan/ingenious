from ingenious.models.agent import Agent, Agents, IProjectAgents
from ingenious.models.config import Config


class ProjectAgents(IProjectAgents):
    def Get_Project_Agents(self, config: Config) -> Agents:
        local_agents = []


        # Submission evaluation agents (new workflow)
        local_agents.append(
            Agent(
                agent_name="submission_evaluator_agent",
                agent_model_name="gpt-5-mini",
                agent_display_name="Submission Evaluator",
                agent_description="Evaluates individual submissions against specific criteria.",
                agent_type="researcher",
                model=None,
                system_prompt=None,
                log_to_prompt_tuner=True,
                return_in_response=False,
            )
        )
        local_agents.append(
            Agent(
                agent_name="criteria_analyzer_agent",
                agent_model_name="gpt-5-mini",
                agent_display_name="Criteria Analyzer",
                agent_description="Analyzes and interprets evaluation criteria for scoring.",
                agent_type="researcher",
                model=None,
                system_prompt=None,
                log_to_prompt_tuner=True,
                return_in_response=False,
            )
        )
        local_agents.append(
            Agent(
                agent_name="scoring_agent",
                agent_model_name="gpt-5-mini",
                agent_display_name="Scoring Agent",
                agent_description="Assigns scores to submissions based on criteria evaluation.",
                agent_type="researcher",
                model=None,
                system_prompt=None,
                log_to_prompt_tuner=True,
                return_in_response=False,
            )
        )

        # Common agents
        local_agents.append(
            Agent(
                agent_name="summary",
                agent_model_name="gpt-5-mini",
                agent_display_name="Summarizer",
                agent_description="Generates summary reports from agent analysis.",
                agent_type="summary",
                model=None,
                system_prompt=None,
                log_to_prompt_tuner=True,
                return_in_response=True,
            )
        )
        local_agents.append(
            Agent(
                agent_name="user_proxy",
                agent_model_name="gpt-5-mini",
                agent_display_name="user_proxy_agent",
                agent_description="Coordinates workflow and message routing.",
                agent_type="user_proxy",
                model=None,
                system_prompt=None,
                log_to_prompt_tuner=False,
                return_in_response=False,
            )
        )

        return Agents(agents=local_agents, config=config)
