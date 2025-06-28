First i initialized project uv init

Then uv venv and activate

then uv add -r requirements.txt and install dependencies.


In langraph_mcp base folder run below command
uv run streamlit run app.py

class DebugTavilySearch:
    def __init__(self, *args, **kwargs):
        self.tool = TavilySearch(*args, **kwargs)
        self.name = getattr(self.tool, 'name', 'TavilySearch')
    def __call__(self, *args, **kwargs):
        print(f"[DEBUG] Calling TavilySearch tool with args: {args}, kwargs: {kwargs}")
        result = self.tool(*args, **kwargs)
        print(f"[DEBUG] TavilySearch result: {result}")
        return result
    def __getattr__(self, name):
        return getattr(self.tool, name)
    @property
    def __name__(self):
        # This makes it look like a function to LangChain
        return getattr(self.tool, '__name__', self.name)