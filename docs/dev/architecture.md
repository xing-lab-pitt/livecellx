## SingleCellStatic  
class designed to hold all information about a single cell at some timepoint
attributes
- time point
- id (optional)
- contour coordinates
- cell bounding box
- img crop (lazy)
- feature map 
- original img (reference/pointer)

## SingleCellTrajectory

## SingleCellTrajectoryCollection

## Expected information after each stage
**Note**  
If you already have satisfying segmentation models or segmentation results, you may skip Annotation and Segmentation part below.

### Annotation
After annotating imaging datasets, you should have json files ready for segmentation training. 

#### labelme
if you use labelme following our annotation protocol, please transform labelme json to COCO format. A fixed version of labelme2coco implementation is included within our package. Please refer to our tutorial.

### Segmentation


### Track

### Trajectory