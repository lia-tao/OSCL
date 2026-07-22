# OSCL
Online Semantic Correlation Learning for Cross-modal Retrieval


- The online training and query data chunks for four dataset (IAPR TC-12, MS-COCO, NUS-WIDE and MIRFlickr): https://drive.google.com/drive/folders/1q-LamOdA_M3J0hnLrjm03qF08ZvmnhBe

- Compared Algorithms:
  - [Multi-view multi-label canonical correlation
analysis for cross-modal multimedia retrieval](https://github.com/asharani97/Fast-MVMLCCA)
  - [Multi-model semantic autoencoder for cross-modal retrieval](https://github.com/yiling2018/mmsae)
  - [Discrete online cross-modal hashing with consistency preservation](https://github.com/XX-kang/DOCMH)
  - [OH-CMH: Towards cross-modal hashing for streaming data with hierarchical labels and label increment scenario](https://github.com/pjunjie/OH-CMH)
  - [Multiple Information Embedded Hashing for Large-Scale Cross-Modal Retrieval](https://github.com/yxinwang/MIEH)
  - [One for more: Structured Multi-Modal Hashing for multiple multimedia retrieval tasks](https://github.com/ChaoqunZheng/SMMH)
  - [Discrete online cross-modal hashing](https://github.com/yw-zhan/DOCH)



In MATLAB: run main_demo.m, will see the summary result table of, for example, MIRFlickr.


   Round    TrainingPairs    ItoT_mAP    TtoI_mAP        AccumulatedTrainingTimeSeconds
    _____    _____________    ________    ________       ______________________________
      1           2000        0.67186     0.76492           5.9508            
      2           4000        0.69372     0.75563           18.072            
      3           6000        0.70541     0.74969           36.697            
      4           8000        0.71049     0.74316           56.941            
      5          10000        0.71629     0.74146           81.454            
      6          12000        0.72109     0.74175           106.45            
      7          14000        0.72393      0.7413           133.09            
      8          16000        0.72564     0.74184           163.73            
      9          18000        0.72573     0.74042           207.39            


