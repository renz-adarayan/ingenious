import json
from typing import List, Union

from pydantic import BaseModel, Field

from ingenious.utils.model_utils import Listable_Object_To_Csv


class RootModel_Bike(BaseModel):
    brand: str
    model: str
    year: int
    price: float


class RootModel_MountainBike(RootModel_Bike):
    suspension: str = Field(
        ..., description="Type of suspension (e.g., full, hardtail)"
    )


class RootModel_RoadBike(RootModel_Bike):
    frame_material: str = Field(
        ..., description="Material of the frame (e.g., carbon, aluminum)"
    )


class RootModel_CustomerReview(BaseModel):
    rating: float
    comment: str


class RootModel_ElectricBike(RootModel_Bike):
    battery_capacity: float = Field(..., description="Battery capacity in kWh")
    motor_power: float = Field(..., description="Motor power in watts")


class RootModel_BikeStock(BaseModel):
    bike: Union[RootModel_MountainBike, RootModel_RoadBike, RootModel_ElectricBike]
    quantity: int


class RootModel_BikeSale(BaseModel):
    product_code: str
    quantity_sold: int
    sale_date: str
    year: int
    month: str
    customer_review: RootModel_CustomerReview


class RootModel_BikeSale_Extended(RootModel_BikeSale):
    store_name: str
    location: str


class RootModel_Store(BaseModel):
    name: str
    location: str
    bike_sales: List[RootModel_BikeSale]
    bike_stock: List[RootModel_BikeStock]


class RootModel(BaseModel):
    stores: List[RootModel_Store]

    @staticmethod
    def load_from_json(json_data: str) -> None:
        data = json.loads(json_data)
        root_model = RootModel(**data)
        print(root_model)

    def display_bike_sales_as_table(self) -> str:
        table_data: list[RootModel_BikeSale_Extended] = []

        for store in self.stores:
            for sale in store.bike_sales:
                store_name = store.name
                location = store.location
                rec = RootModel_BikeSale_Extended(
                    store_name=store_name, location=location, **sale.model_dump()
                )
                table_data.append(rec)

        ret = Listable_Object_To_Csv(table_data, RootModel_BikeSale_Extended)
        # Note always provide tabular data with a heading as this allows our datatables extension to render the data correctly
        return "## Sales\n" + ret
