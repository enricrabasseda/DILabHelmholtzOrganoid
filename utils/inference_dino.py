# File containing all functions for DINO inference
import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import annotate, predict

# Define IoU
def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union (IoU) between a box and a set of boxes.

    Args:
        box (torch.Tensor): Bounding box [4] [xtopleft, ytopleft, xbottomright, ybottomright].
        boxes (torch.Tensor): Set of bounding boxes [N,4], where each row is [xtopleft, ytopleft, xbottomright, ybottomright].

    Returns:
        torch.Tensor: IoU values [N].
    """
    # Coordinates of the intersection rectangle
    intersection_xtl = torch.maximum(box[0], boxes[:, 0])
    intersection_ytl = torch.maximum(box[1], boxes[:, 1])
    intersection_xbr = torch.minimum(box[2], boxes[:, 2])
    intersection_ybr = torch.minimum(box[3], boxes[:, 3])

    # Area of the intersection rectangle
    intersection_area = torch.max(torch.tensor(0.0), intersection_xbr - intersection_xtl) * torch.max(torch.tensor(0.0), intersection_ybr - intersection_ytl)

    # Area of the union of the two rectangles
    union_area = (box[2] - box[0])*(box[3] - box[1]) + (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1]) - intersection_area

    # IoU: Intersection over Union
    iou = intersection_area / union_area

    return iou

def area(boxes):
    """
    Compute the area of given boxes

    Args:
        boxes (torch.Tensor): Bounding boxes [N,4], where each row is [xtopleft, ytopleft, xbottomright, ybottomright].
    Returns:
        torch.Tensor: area values [N]
    """
    ### Compute the area of the boxes
    final_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])

    return final_area

def overlapping_boxes(boxes, stride_distance = 0.02):
    """
    Given boxes it searches for which ones there is an overlapping of boxes such
    that they contain small boxes

    Args:
        boxes (torch.Tensor): Bounding boxes [N,4], where each row is [xtopleft, ytopleft, xbottomright, ybottomright].
        stride_distance (float): distance to move the big box to reach a bigger set of possible output boxes.
        This allows to take boxes that are not only entirely contained on big boxes but almost contained.

    Returns:
        torch.Tensor: Indices of selected bounding boxes after the filtering [S].
    """

    # First we compute the surface of the boxes
    surface = area(boxes)

    # Get the indices after ordering the boxes by their surface in decreasing order
    sorted_indices = torch.argsort(surface, descending=True)

    include_indices = []

    for big_index, big_box in enumerate(boxes):
            # Exclude the current big box from the set of all boxes
            small_boxes = torch.cat((boxes[:big_index], boxes[big_index+1:]))

            # Compute ious
            ious = calculate_iou(big_box, small_boxes)

            # Check overlapping for all boxes that intersect (iou > 0) with the
            # big one with stride distance
            if torch.any(ious > 0):
                small_boxes = small_boxes[ious > 0]
                colapse = (torch.maximum(big_box[0] - stride_distance, small_boxes[:, 0]) == small_boxes[:, 0]) & (torch.maximum(big_box[1] - stride_distance, small_boxes[:, 1]) == small_boxes[:, 1]) & (torch.minimum(big_box[2] + stride_distance, small_boxes[:, 2]) == small_boxes[:, 2]) & (torch.minimum(big_box[3] + stride_distance, small_boxes[:, 3]) == small_boxes[:, 3])
                # If there is not any box entirely contained in the big one, we consider it
                if colapse.sum() == 0:
                  include_indices.append(big_index)
            # If it does not overlap with any box we consider it
            else:
              include_indices.append(big_index)

    return torch.tensor(include_indices)


def nms(boxes, scores, threshold_nms=0.5):
    """
    Perform Non-Maximum Suppression to filter out overlapping boxes with lower confidence scores.

    Args:
        boxes (torch.Tensor): Bounding boxes [N,4], where each row is [xtopleft, ytopleft, xbottomright, ybottomright].
        scores (torch.Tensor): Confidence scores for each bounding box [N].
        threshold_nms (float): IoU threshold to determine overlapping boxes.

    Returns:
        torch.Tensor: Indices of selected bounding boxes after NMS [S].
    """
    # Sort bounding boxes by their scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    selected_indices = []

    while len(sorted_indices) > 0:
        # Pick the box with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Calculate IoU (Intersection over Union) with other boxes
        ious = calculate_iou(boxes[current_index], boxes[sorted_indices[1:]])

        # Keep only boxes with IoU below the threshold
        # below_thresholds = ious <= threshold
        below_threshold = ious <= threshold_nms
        sorted_indices = sorted_indices[1:][below_threshold]

    return torch.tensor(selected_indices)

# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25,
           NMS = False, threshold_nms=0.5, stride_distance = 0.02):
  """
  Detect all the relevant boxes for an image and delete all boxes that contain
  repeated information given by other ones.

  Args:
      image (torch.Tensor): Tensor containing the image [C, W, H]
      image_source (np.array): Array containing the original image [W, H, C]
      text_prompt (string): Descriptions of the objects to be searched in the image.
      model: to be imported from Grounding DINO.
      box_threshold (float): value in (0,1) with minimum score of a box to be showed.
      text_threshold (float): value in (0,1) with minimum similarity of a text to box to be showed
      NMS (boolean): True if NMS algorithm should be applied
      threshold_nms (float): value in (0,1) with minimum value of iou to consider that two boxes intersect in NMS
      stride_distance (float): distance to move the big box to reach a bigger set of possible output boxes.

  Returns:
      annotated_frame (numpy.array): image containing all boxes found (W, H, C) and a identifier.
      boxes_filtered (torch.Tensor): Bounding boxes [N,4], where each row is [xtopleft, ytopleft, xbottomright, ybottomright].
  """
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  # Convert boxes to xyxy format since it is the only compatible format for our boxes
  boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

  # Get rid of big overlapping boxes
  selected_indices = overlapping_boxes(boxes = boxes_xyxy,
                                       stride_distance = stride_distance)
  prev = boxes.shape[0]
  boxes_xyxy_filtered = boxes_xyxy[selected_indices]
  boxes_filtered = boxes[selected_indices]
  logits_filtered = logits[selected_indices]
  print("Filtered number of boxes was", prev, ", after Big Box Overlap Filtering is", boxes_filtered.shape[0])

  # Apply Non-Maximum Supremum if it is specified
  if NMS:
    selected_indices = nms(boxes = boxes_xyxy_filtered,
                           scores = logits_filtered,
                           threshold_nms=threshold_nms)
    prev = boxes_xyxy_filtered.shape[0]
    boxes_xyxy_filtered = boxes_xyxy_filtered[selected_indices]
    boxes_filtered = boxes_filtered[selected_indices]
    logits_filtered = logits_filtered[selected_indices]
    print("Filtered number of boxes was", prev, ", after NMS is", boxes_filtered.shape[0])

  # Define an identifier for each of the boxes
  ids = range(len(logits_filtered))

  # Annotate using the given function of Grounding DINO library
  annotated_frame = annotate(image_source=image_source, boxes=boxes_filtered, logits=logits_filtered, phrases=ids)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB
  return annotated_frame, boxes_filtered