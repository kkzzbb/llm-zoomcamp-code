import sqlite3
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from starter import RAGBase, rag as starter_rag

load_dotenv()


class SQLiteSpanExporter(SpanExporter):
	def __init__(self, db_path="traces.db"):
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS spans (
					name TEXT,
				start_time INTEGER,
				end_time INTEGER,
				input_tokens INTEGER,
				output_tokens INTEGER,
				cost REAL
			)
		""")
		self.conn.commit()
	
	def export(self, spans):
		for span in spans:
			attrs = dict(span.attributes or {})
			self.conn.execute(
				"INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
				(
					span.name,
					span.start_time,
					span.end_time,
					attrs.get("input_tokens"),
					attrs.get("output_tokens"),
					attrs.get("cost"),
				),
			)
		self.conn.commit()
		return SpanExportResult.SUCCESS

	def shutdown(self):
		self.conn.close()

	def force_flush(self):
        	return True

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)

provider = TracerProvider()
# For Q1 and Q2: Uncomment the line below and comment out the SQLite exporter
# provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")

def calculate_cost(input_tokens, output_tokens):
	return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

class RAGTraced(RAGBase):
	def rag(self, query):
		with tracer.start_as_current_span("rag"):
			return super().rag(query)
	
	def search(self, query, num_results=5):
		with tracer.start_as_current_span("search"):
			return super().search(query, num_results)
		
	def llm(self, prompt):
		with tracer.start_as_current_span("llm") as span:
			response = super().llm(prompt)
			       
			usage = response.usage

			in_tokens = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', 0))
			out_tokens = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', 0))
			
			span.set_attribute("input_tokens", in_tokens)
			span.set_attribute("output_tokens", out_tokens)
			
			span.set_attribute("cost", calculate_cost(in_tokens, out_tokens))
			return response
	
traced_rag = RAGTraced(
	index=starter_rag.index,
	llm_client=starter_rag.llm_client
)

query = "How does the agentic loop keep calling the model until it stops?"
answer = traced_rag.rag(query)
print("\n--- Answer ---")
print(answer)