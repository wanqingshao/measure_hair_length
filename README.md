# Measuring hair length

This repo contains the scripts used for automatic hair measurement. Project is under development.

The app takes in images containing mouse hair, identifies and highlights individual hair, then outputs the length of each hair.

## App overview

GUI was built with `PyQt5`. User could select input and output folder. Objects smaller than "minimum length of consider (pixels)" will be filtered out. By default, objects touching the edge of the image will be filtered out.

<p align="center">
  <img src="https://github.com/wanqingshao/measure_hair_length/blob/main/app_overview/overview.png" width="400" alt="app overview">
</p>

## Output overview

The app skeletonizes the input image with `skimage` and summarizes the skeleton stats with `skan`. Original, grey and skeletonized images were output. To resolve overlapping and broken hairs, the entry and exit angles of each path are calculated and paths with the smallest angle change less than 80 degree are chained together. Chained paths with overlapping components are then prioritized based on the average angle change. Resolved hair objects that pass the minimum length threshold and satisfy the edge status selection are highlighted and their length is recorded in the output spreadsheet.

Test image and corresponding output are included in `test_input_image` and `test_output_image`

<p align="center">
  <img src="https://github.com/wanqingshao/measure_hair_length/blob/main/app_overview/overview2.png" width="1000" alt="output overview">
</p>
