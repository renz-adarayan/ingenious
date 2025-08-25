# Custom Data Models

This directory contains **Pydantic data models** that define the structure of your data and ensure type safety throughout your AI workflows.

## 🎯 **Purpose**

Custom models serve multiple critical functions:
- **🔒 Data Validation** - Ensure incoming data matches expected structure
- **📝 Type Safety** - Catch errors early with strong typing
- **🔄 Serialization** - Convert between JSON, CSV, and Python objects
- **📖 Documentation** - Self-documenting code with clear data contracts


## 📊 **Example: Bike Sales Models (`bike_insights/bikes.py`, `bike_insights/agent.py`)**

The template includes a complete set of models for the bike insights workflow, now located in `models/bike_insights/bikes.py` and `models/bike_insights/agent.py`:

### **🚴 Core Models**
```python
class RootModel_Bike(BaseModel):
    brand: str
    model: str
    year: int
    price: float

class RootModel_ElectricBike(RootModel_Bike):
    battery_capacity: float = Field(..., description="Battery capacity in kWh")
    motor_power: float = Field(..., description="Motor power in watts")

class RootModel_CustomerReview(BaseModel):
    rating: float  # 1.0 to 5.0
    comment: str
```

### **🏪 Store and Sales Models**
```python
class RootModel_BikeSale(BaseModel):
    product_code: str
    quantity_sold: int
    sale_date: str
    year: int
    month: str
    customer_review: RootModel_CustomerReview

class RootModel_Store(BaseModel):
    name: str
    location: str
    bike_sales: List[RootModel_BikeSale]
    bike_stock: List[RootModel_BikeStock]
```

### **🌟 Root Data Container**
```python
class RootModel(BaseModel):
    stores: List[RootModel_Store]

    def display_bike_sales_as_table(self):
        """Convert sales data to CSV format for AI agents"""
        # Implementation details...
        return "## Sales\n" + csv_data
```

## 🛠️ **Creating Your Own Models**

### **1. Define Your Domain Objects**
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime

class YourCustomModel(BaseModel):
    # Required fields
    id: str
    name: str
    created_at: datetime

    # Optional fields with defaults
    status: str = "active"
    metadata: Optional[dict] = None

    # Validated fields with constraints
    score: float = Field(..., ge=0, le=100, description="Score between 0-100")

    # Custom validation
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

### **2. Model Best Practices**
```python
class ProductModel(BaseModel):
    """✅ Good practices demonstrated"""

    # Clear, descriptive field names
    product_id: str = Field(..., description="Unique product identifier")
    display_name: str = Field(..., min_length=1, max_length=200)

    # Proper typing
    price: float = Field(..., gt=0, description="Price in USD")
    tags: List[str] = Field(default_factory=list)

    # Enums for constrained values
    category: ProductCategory  # Define enum separately

    # Nested models for complex data
    specifications: ProductSpecs

    # Helper methods
    def to_display_format(self) -> str:
        """Convert to human-readable format"""
        return f"{self.display_name} - ${self.price:.2f}"

    def to_csv_row(self) -> dict:
        """Convert to flat dictionary for CSV export"""
        return {
            "id": self.product_id,
            "name": self.display_name,
            "price": self.price,
            "category": self.category.value
        }

# ❌ Avoid these patterns:
class BadModel(BaseModel):
    data: dict  # Too generic
    x: str      # Unclear naming
    # No validation or documentation
```

### **3. Integration with AI Workflows**
```python
class ConversationPayload(BaseModel):
    """Model for AI workflow input"""

    # Required workflow metadata
    revision_id: str
    identifier: str
    conversation_flow: str

    # Your domain data
    business_data: YourCustomModel

    # Optional context
    user_context: Optional[dict] = None
    previous_results: Optional[List[dict]] = None

    def prepare_for_agents(self) -> str:
        """Convert to format suitable for AI agents"""
        return json.dumps({
            "metadata": {
                "revision_id": self.revision_id,
                "identifier": self.identifier
            },
            "data": self.business_data.dict(),
            "context": self.user_context or {}
        }, indent=2)
```

## 🔄 **Data Transformation Utilities**

### **CSV Export Functions**
```python
from ingenious.utils.model_utils import Listable_Object_To_Csv

class ReportModel(BaseModel):
    products: List[ProductModel]

    def to_csv_report(self) -> str:
        """Generate CSV report for AI analysis"""
        csv_data = Listable_Object_To_Csv(
            self.products,
            ProductModel
        )
        return "## Product Analysis\n" + csv_data

    def get_summary_stats(self) -> dict:
        """Calculate summary statistics"""
        return {
            "total_products": len(self.products),
            "avg_price": sum(p.price for p in self.products) / len(self.products),
            "categories": list(set(p.category for p in self.products))
        }
```

### **JSON Schema Generation**
```python
# Generate JSON schema for API documentation
schema = YourCustomModel.schema()
print(json.dumps(schema, indent=2))

# Use for validation in other languages
with open('schemas/your_model.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

## 🧪 **Testing Your Models**

### **Unit Tests**
```python
import pytest
from pydantic import ValidationError

def test_product_model_validation():
    # ✅ Valid data
    valid_data = {
        "product_id": "BIKE-001",
        "display_name": "Electric Mountain Bike",
        "price": 2500.0,
        "category": "electric"
    }
    product = ProductModel(**valid_data)
    assert product.price == 2500.0

    # ❌ Invalid data
    with pytest.raises(ValidationError):
        ProductModel(product_id="", price=-100)  # Empty ID, negative price

def test_data_transformation():
    product = ProductModel(**valid_data)
    csv_row = product.to_csv_row()
    assert "BIKE-001" in csv_row["id"]
    assert csv_row["price"] == 2500.0
```

### **Integration Tests**
```python
def test_workflow_integration():
    """Test model with actual AI workflow"""

    # Create test payload
    payload = ConversationPayload(
        revision_id="test-123",
        identifier="test-run-456",
        conversation_flow="product_analysis",
        business_data=ProductModel(**test_data)
    )

    # Test serialization
    json_str = payload.prepare_for_agents()
    assert "test-123" in json_str

    # Test deserialization
    loaded = ConversationPayload.parse_raw(payload.json())
    assert loaded.revision_id == "test-123"
```

## 📋 **Model Organization Patterns**


### **Single Domain File (Before)**
```
models/
├── bikes.py          # All bike-related models
├── agent.py          # Bike agent logic
├── customers.py      # Customer and review models
├── analytics.py      # Reporting and metrics models
└── __init__.py       # Import organization
```



### **Bike Insights Workflow**
```
models/
├── bike_insights/
│   ├── bikes.py          # All bike-related models
│   ├── agent.py          # Bike agent logic
│   ├── customers.py      # Customer and review models
│   ├── analytics.py      # Reporting and metrics models
│   └── __init__.py       # Import organization
│   └── ...
```

---

**🗂️ Custom Workflow Organization**

Each custom workflow should have its own folder inside `models/`, like `bike_insights`. This keeps related models, agents, and logic organized and makes it easier to maintain and extend workflows. For example:

```
models/
├── bike_insights/
│   ├── bikes.py
│   ├── agent.py
│   └── ...
├── restaurant_recommender/
│   ├── restaurants.py
│   ├── agent.py
│   └── ...
├── project_management/
│   ├── projects.py
│   ├── agent.py
│   └── ...
```

This pattern ensures each workflow is modular and self-contained.



### **Modular Structure (Example: bike_insights)**
```
models/
├── bike_insights/
│   ├── base/
│   │   ├── __init__.py
│   │   ├── common.py     # Shared base classes
│   │   └── enums.py      # Common enumerations
│   ├── domain/
│   │   ├── bikes.py      # Product models
│   │   ├── sales.py      # Sales transaction models
│   │   └── users.py      # User profile models
│   ├── workflows/
│   │   ├── payloads.py   # AI workflow input models
│   │   └── responses.py  # AI workflow output models
│   ├── agent.py
│   ├── customers.py
│   ├── analytics.py
│   └── __init__.py
├── restaurant_recommender/
│   ├── ...
├── project_management/
│   ├── ...
```

## 🔗 **Integration Points**

### **With AI Agents**
```python
# In your conversation flow
from models.bikes import RootModel

async def process_bike_data(chat_request: ChatRequest):
    # Parse incoming data with your model
    bike_data = RootModel.parse_raw(chat_request.user_prompt)

    # Use model methods for data preparation
    table_data = bike_data.display_bike_sales_as_table()

    # Pass structured data to agents
    await agent.process(table_data)
```

### **With API Endpoints**
```python
# In custom API routes (../api/)
from fastapi import APIRouter
from models.bikes import RootModel, ProductModel

router = APIRouter()

@router.post("/analyze-products")
async def analyze_products(products: List[ProductModel]):
    # Automatic validation via Pydantic
    # Type hints provide IDE support
    total_value = sum(p.price for p in products)
    return {"total_value": total_value, "count": len(products)}
```

## 📚 **Related Documentation**

- **🌊 Workflows:** See `../services/README.md` for using models in workflows
- **📊 Sample Data:** See `../sample_data/README.md` for test data structures
- **🧪 Testing:** See `../tests/README.md` for testing patterns
- **📖 Pydantic Docs:** [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

---

**💡 Key Takeaway:** Well-designed models are the foundation of reliable AI workflows. They catch errors early, document your data contracts, and make your code more maintainable.
