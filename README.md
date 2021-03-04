# Dynamic-Foot-Pressure-Recognition
---
[ASVDA - 動態足壓影像辨識](https://aidea-web.tw/topic/d6d8b111-d915-4ea9-89ee-43e148c37f6e)

## 設備與環境
- 硬體：
  - CPU：i7-9700
  - GPU：RTX 2070
- 軟體：
  - Cuda：cuda_**11.0**.2_451.48_win10
  - cuDNN：cudnn-11.0-windows-x64-v**8.0.4**.30
  - TensorFlow：tensorflow-**2.4**.0

## 構思與過程
- 第一次接觸這種題目，不太確定是否屬於 ComputerVision 領域，也不知道該用何種類似題目去 Google、PaperWithCode 和 Github 搜尋，所以除了經典的深層卷積網路，我嘗試自製了一個裁切圖片並運算的深層卷積網路，但並沒有使用任何論文的架構。
- 關於資料前處理：
  1. 由於圖片有左右腳分別，初步構想就是想規格化所有圖片，因此我對所有圖片進行裁切，獨立出 6×19 的數組區塊（像素數量為 102×316）。
  2. 由於圖片中有數字跟顏色，我認為可以嘗試各自進行辨識，所以我做了兩件事：
     1. 找出數字的排列規律：在上一步的裁切中，像素寬度 102 除以 6 個數組後得出 17 並不能整除單數組共 3 位數字令人十分疑惑，但在仔細觀察後發現每個三位數的排列方式皆是 14141141，其中 1 是邊緣、4 是重點部分（參照下圖），因此可確認 17 確實是正確的單數組寬度；對於高度則較無特別之處，重點部分高度為 8、邊緣為 1，故高度整體為 10。結論：單一數組之尺寸為 17×10。
        > ![](https://i.imgur.com/3ujDIjw.png)
     2. 獨立出顏色：在競賽討論區可看到此壓力圖之 Colormap 並非公式，雖可透過自製 Colormap 並將原圖轉換為純粹的壓力數值，但我時間不夠，所以沒有進行嘗試，不然我認為這會是非常強大的 Input。我做的是將整張圖每個像素為黑色或灰色的部分（數字部分）以上方那一格的顏色替換，並在完成全部後做一次模糊（PIL.ImageFilter.BLUR），然後以此圖做為其中一個 Input。
- 關於經典的深層卷積網路：
  我將三張圖片作為 Input：原圖、裁切後以黑色補齊外框的圖、色彩獨立圖（資料前處理-2.2）。先將三張圖 concatenate 成 (BATCH_SIZE, 120, 400, 9) 的矩陣作為 Input 後，做三層卷積與 MaxPooling2D 將其尺寸縮小至 (BATCH_SIZE, 15, 50, N)，然後做 Flatten、Dropout、BatchNormalization，再進行數層的 Fully connected layers 直至將尺寸縮小到 (BATCH_SIZE, 4)。
- 關於自製的深層卷積網路：
  我一樣將三張圖片作為 Input：原圖、裁切後以黑色補齊外框的圖、色彩獨立圖（資料前處理-2.2）。但不同的是，在將三張圖輸入之後，模型會先將「裁切後以黑色補齊外框的圖」的黑色邊框去除，使其變回原本的 102×316，然後做兩個不同的濾鏡（PIL.ImageFilter.SHARPEN、PIL.ImageFilter.CONTOUR）；至此，已經有 5 張圖可以做為 Input 了。
  > P.s. 這邊是很自由的，我也可以對色彩獨立圖做濾鏡，並且也還有很多濾鏡可以選擇，也可以反覆套用疊加。
  並且我定義了三種 Block：一是給 120×400 的經典卷積；二是給 102×316 的經典卷積；三是會將每個數組（17×10）和數字（6×10）切分並各自餵入卷積和一次 MaxPooling2D（17×10 → 9×5、6×10 → 3×5），各自 Flatten、Dropout、BatchNormalization、Fully connected layers 後結合再做一次一模一樣的動作。
  最後將 Inputs 各自代入合適的其中一個 Block，得出矩陣後將所有矩陣 concatenate、Flatten、Dropout、BatchNormalization、Fully connected layers 直至最後剩下 (BATCH_SIZE, 4) 為輸出為止。

<!--
## 結果
- 得到競賽第 9 名。
- 第 1 名使用的架構在[此篇論文](https://paperswithcode.com/paper/feature-learning-for-chord-recognition-the)中。
- 第二名與我使用相同架構，但資料提取、前處理與後處理都差很多，所以成績也很差多。
-->

## 心得
1. 在做資料前處理時，若使用 os.listdir(DIRECTORY) 讀取某資料夾之檔案，須注意讀取順序可能跟想像得不一樣。
2. 與上次的[和弦辨識競賽](https://github.com/aisu-programming/Chord-Estimation)不同，這次的模型不是輸出分類，而是輸出「數值」；應該早點確認 Output 是否有奇怪的地方。（例如這次就是所有 BATCH 的 x1, y1, x2, y2 都輸出了 Loss 最低的統一值，難怪 Loss 卡住降不下去。）
3. 設計自創模型時應注意計算效率，並且日後應該搜尋一下平行化演算的方法。（本次的 CustomizedConvolutionModel 迭代一次需要近 2 分鐘，難以訓練。）