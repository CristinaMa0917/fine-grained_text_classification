# fine-grained_text_classification
Intern work in Alibaba: Classification of search query , altogether 13k query types(categories).In fact, there is a mapping from search query in Taobao to shopping categories. 

The categories are built like a tree. And the parent categories are like women's clothing or men's clothing. The parent categories are divided into fine-grained categories which are called son categories. The total num of son categories are around 13k (12764).
![image](https://github.com/CristinaMa0917/fine-grained_text_classification/blob/SWEM/images/img1.png)

The main contribution of my work is the modeling of the parent-son categories prediction process. The image as following shows the framework of PS-SWEM. And the final prediction accuracy of PS-SWEM is 69.35%.
![image](https://github.com/CristinaMa0917/fine-grained_text_classification/blob/SWEM/images/imgs2.png)

