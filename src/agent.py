import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ResearchAgent:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Check your .env file.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

    def write_report(self, data_dict):
        """
        Synthesizes the schema data into a readable narrative.
        data_dict: The dictionary version of StockDataSchema.
        """
        print(" Gemini is analyzing data and writing the report...")
        
        template = """
        You are a Senior Equity Research Analyst. Write a one-page research update.
        
        ### DATA SNAPSHOT
        - Ticker: {ticker}
        - Current Price: {currency_symbol}{current_price}
        - AI Prediction (30 Days): {currency_symbol}{prediction_30d}
        - Trend Forecast: {trend}
        - RSI (14): {rsi}
        - SMA 50: {sma_50}
        - P/E Ratio: {pe_ratio}
        
        ### NEWS CONTEXT
        {news_summary}
        
        ### INSTRUCTIONS
        Write a structured report with these sections:
        1. **Executive Summary**: A concise recommendation based on data.
        2. **Technical Analysis**: Interpret the RSI and SMA vs Price.
        3. **Future Outlook**: Discuss the AI prediction ({trend}) and what it implies.
        4. **Risks & News**: Summarize key headlines and potential risks.
        
        Keep it professional, objective, and under 500 words.
        """
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke(data_dict)