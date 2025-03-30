from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType     
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer  
import os
import pydicom
import numpy as np
from qdrant_client.http.models import PointStruct
import torch
import cv2
from PIL import Image
from torchvision import models
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType     
from qdrant_client.http.models import Batch, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel

import torchvision.transforms as transforms

class Server():
    def __init__(self):
        """Qdrant sunucusunu başlatır ve istemciyi oluşturur."""
        self.client = QdrantClient("http://localhost:6333")
        self.txtModel = SentenceTransformer('all-MiniLM-L6-v2')
        self.processor = AutoProcessor.from_pretrained("microsoft/resnet-50")
        self.model = AutoModel.from_pretrained("microsoft/resnet-50")
        self.originalCount = 0
        self.filteredCount = 0
        self.reportCount = 0

        """self.client.create_payload_index(
            collection_name="dicom_files",
            field_name="name",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        self.client.create_payload_index(
            collection_name="dicom_files",
            field_name="data",
            field_schema=PayloadSchemaType.TEXT,
        )"""
    
    """def start_server(self, path="./qdrant_data"):
        "Qdrant sunucusunu başlatmak için bir terminal komutunu çağırır."
        os.system(f"qdrant --storage {path}")
        print(f"Qdrant server started with storage path: {path}")"""
    
    def create_collection(self):
        """for i, filter in enumerate(filters):
            if i == 2:
                return
            c_name = str(filter) + "_embeddings"
            if self.client.collection_exists(collection_name=c_name) == False:
                self.client.create_collection(  
                    collection_name=c_name,
                    vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
                )"""

  
        if self.client.collection_exists(collection_name="originalImages") == False:
            self.client.create_collection(  
                collection_name="originalImages",
                vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
            )
        if self.client.collection_exists(collection_name="filteredImages") == False:
            self.client.create_collection(  
                collection_name="filteredImages",
                vectors_config=None,
            )
        if self.client.collection_exists(collection_name="radReport") == False:
            self.client.create_collection(  
                collection_name="radReport",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
    
    def add_point(self, idlist, vectorlist, payloadlist, name):
        print(name)
        """self.client.upsert(
            collection_name=f"{name}_embeddings",
            points=m.Batch(    
                ids=idlist, 
                vectors=vectorlist, 
                payloads=payloadlist,
            ),
        )"""
        self.client.upsert(
            collection_name="originalImages",
            points=Batch(    
                ids=idlist, 
                vectors=vectorlist, 
                payloads=payloadlist,
            ),
        )
        self.originalCount += len(idlist)

    def add_filtered_point(self, idx, payloadx):
        self.client.upsert(
            collection_name="filteredImages",
            points=[
                PointStruct(
                    id=idx,
                    payload=payloadx,
                    vector={},
                ),
            ],
        )

    def add_radReport_point(self, idx, vectorx, payloadx):
        self.client.upsert(
            collection_name="radReport",
            points=[
                PointStruct(
                    id=idx,
                    payload=payloadx,
                    vector=vectorx,
                ),
            ],
        )

    def update_radReport_point(self, idx, payloadx):
        """self.client.upsert(
            collection_name="radReport",
            points=[
                PointStruct(
                    id=idx,
                    payload=payloadx,
                ),
            ],
        )"""

        self.client.set_payload(
            collection_name="radReport",
            payload={
                "report": payloadx,
            },
            points=[idx],
        )

    def search(self, name, id):
        """collection_info = self.client.get_collection(collection_name="Orijinal_embeddings")

        # Koleksiyondaki toplam point sayısını alın
        points_count = collection_info.points_count
        return points_count
        print(f"Koleksiyon '{collection_name}' toplam {points_count} point içeriyor.")"""
        results = self.client.scroll(
            collection_name="originalImages",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="name", match=MatchValue(value=name)),
                    FieldCondition(key="id", match=MatchValue(value=id)),
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        return results

    def search_filtered(self, id, name):
        results = self.client.scroll(
            collection_name="filteredImages",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="id", match=MatchValue(value=id)),
                    FieldCondition(key="filter", match=MatchValue(value=name)),
                ]
            ),
            limit=30,
            with_payload=True,
            with_vectors=False,
        )
        return results
    
    def search_report(self, id):
        results = self.client.scroll(
            collection_name="radReport",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="p_id", match=MatchValue(value=id))
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return results
        
        
    def load_dicom_image(self, pixel_array):
        # Normalize pixel values to 0-255
        pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)
        # Convert to RGB if needed
        if len(pixel_array.shape) == 2:  # Grayscale image
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(pixel_array)


    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).pooler_output
        return embeddings.squeeze().numpy()
        
    def transform_vector(self, vector):
        im = self.load_dicom_image(vector)
        embedding = self.get_image_embedding(im)
        return embedding
    
    def transform_txtVector(self, vector):
        embedding = self.txtModel.encode(vector)
        return embedding
    
    
    
    def create_filter(self, field, value):
        """Belirli bir alan ve değere göre filtre oluşturur."""
        return Filter(
            must=[
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            ]
        )