from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
from PIL import Image

class BaseTask(ABC):
    def __init__(self, dataset_name: str, data_dir: str):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.query_data = self.load_query_data()
        self.support_data = self.load_support_data()
    
    def load_query_data(self) -> List[Dict]:
        query_file = os.path.join(self.data_dir, self.dataset_name, 'query.json')
        with open(query_file, 'r') as f:
            return json.load(f)
    
    def load_support_data(self) -> List[Dict]:
        support_file = os.path.join(self.data_dir, self.dataset_name, 'support.json')
        with open(support_file, 'r') as f:
            return json.load(f)
    
    def load_image(self, image_path: str) -> Image.Image:
        full_path = os.path.join(self.data_dir, image_path)
        return Image.open(full_path).convert('RGB')
    
    @abstractmethod
    def get_task_instruction(self, mode="constrained") -> str:
        pass
    
    @abstractmethod
    def format_demonstration(self, support_item: Dict, include_image_token=True, mode="constrained") -> str:
        pass
    
    @abstractmethod
    def select_demonstrations(self, query: Dict, n_shot: int, seed=None) -> List[Dict]:
        pass
    
    @abstractmethod
    def format_query(self, query: Dict, include_image_token=True, mode="constrained") -> str:
        pass
    
    @abstractmethod
    def evaluate_response(self, query: Dict, response: str, mode="constrained") -> bool:
        pass