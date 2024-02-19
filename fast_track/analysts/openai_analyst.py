""" OpenAI Analyst class. """

from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .analyst import Analyst

load_dotenv()


class OpenAIAnalyst(Analyst):

    def __init__(self,
                 db_uri: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """ Initializes OpenAIAnalyst objects.

        Args:
            db_uri: The database URI.
            model: The model to use for the analyst.
            temperature: The sampling temperature, between 0 and 1.
                Higher values like 0.8 will make the output more random,
                while lower values like 0.2 will make it more focused and
                deterministic. If set to 0, the model will use log probability
                to automatically increase the temperature until certain thresholds
                are hit.
        """
        super().__init__()

        self.db_uri = db_uri
        self.model = model
        self.temperature = temperature

        self.db = None
        self.agent_executor = None

        self._link_db()
        self._create_db_agent()

    @property
    def system_message(self) -> str:
        """ Returns the system message for the analyst.

        Notes:
            Current token count: 323.
        """
        return """
        You are a cutting-edge data analysis and tracking assistant powered by advanced Large Language Models (LLM).
        Your primary focus is on aiding users in the comprehensive analysis, detection, and tracking of data from their
        databases.

        Your Capabilities:

        1. Data Analysis Expertise: Leverage your language understanding capabilities to assist users in deciphering
           intricate patterns, correlations, and insights within their datasets.

        2. Anomaly Detection Mastery: Utilize sophisticated algorithms to excel in identifying anomalies, irregularities, or
           specific events embedded within the dataset. Enhance your user's ability to spot critical information efficiently.

        3. Efficient Tracking Mechanisms: Implement effective tracking systems to monitor changes and movements within the
           database. Keep a vigilant eye on dynamic data elements and provide timely updates to the user.

        Your Role:
        - Query Interpreter: Process user queries in natural language and offer valuable insights based on the principles of
          data analysis, detection, and tracking.
        - Real-time Monitoring Assistant: Act as a vigilant companion, keeping users informed about changes in their data
          through real-time tracking functionalities.

        Guidance for Users:
        - Encourage users to provide detailed queries to extract optimal insights from the data.
        - Emphasize the flexibility of experimenting with different questions and scenarios to unlock the full
          potential of your capabilities.
        - Your mission is to empower users on their data analysis journey, offering them a seamless and intelligent
          experience. Dive into interactions and assist users in uncovering valuable insights from their databases.
        """

    def _create_db_agent(self) -> None:
        """ Creates a LangChain database agent. """
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )
        self.agent_executor = create_sql_agent(
            llm=llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True,
            system_message=self.system_message,
            agent_executor_kwargs={"system_message": self.system_message}
        )

    def _link_db(self) -> None:
        """ Sets up the database connection. """
        self.db = SQLDatabase.from_uri(self.db_uri)

    def query_analyst(self, user_input: str) -> str:
        """ Send a query to analyst.

        Args:
            user_input: text prompt provided by the user.

        Returns:
            a text response from the analyst.
        """
        return self.agent_executor.invoke(input=user_input)
