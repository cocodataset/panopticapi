# COCO 2018 Panoptic Segmentation Task API (Beta version)
This API is an experimental version of [COCO 2018 Panoptic Segmentation Task API](http://cocodataset.org/#panoptic-2018).

## Summary
- **evaluation.py** script calculates [PQ metrics](http://cocodataset.org/#panoptic-eval).

- **format_converter.py** script converts *2 channels PNG panoptic format* (for each pixel first 2 channels of the PNG encode semantic label and instance id respectively) to [*COCO format*](http://cocodataset.org/#format-results).

- **instance_data.py** script extracts things annotations from panoptic ground truth and saves it in [COCO instance segmentation format](http://cocodataset.org/#format-data).

- **semantic_data.py** script extracts semantic segmentation annotation for stuff and things categories from panoptic ground truth. It saves the semantic segmentation as a single channel PNG.

- **visualization.py** script provides an example of visualization for panoptic segmentation data.

## Contact
If you have any questions regarding this API, please contact us at alexander.n.kirillov-at-gmail.com.
