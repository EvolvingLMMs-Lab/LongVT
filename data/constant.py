from enum import Enum

VIDEO_CAPTION_PROMPT = (
    "You are a helpful assistant that summarizes the content of a video. "
    "Please provide a detailed description of the video. "
    "When describing the video, please include the following information: \n"
    "1. The main events in the video. \n"
    "2. The main characters in the video. \n"
    "3. The main locations in the video. \n"
    "4. The main objects in the video. \n"
    "5. The main actions in the video. \n"
    "6. The main emotions in the video. \n"
    "\nMake sure to describe the video in a way that is easy to understand and follow. "
    "Please include as much detail as possible and do not miss any information. "
)


MERGE_EVENT_PROMPT = (
    "You are a helpful assistant that merges events in a video. "
    "Please judge if two consecutive video segments describe the **same event**. "
    "Please answer 'yes' or 'no'.\n"
    "You will be given two captions of consecutive video segments."
    "Each of the caption may contain following information: \n"
    "1. The main events in the video. \n"
    "2. The main characters in the video. \n"
    "3. The main locations in the video. \n"
    "4. The main objects in the video. \n"
    "5. The main actions in the video. \n"
    "6. The main emotions in the video. \n"
    "Please carefully compare the two captions and only answer 'yes' or 'no' in your final response."
)

MERGE_CAPTION_PROMPT = (
    "You are a helpful assistant that merges captions of consecutive video segments. "
    "Please merge the two captions into a single caption. "
    "You will be given two captions of consecutive video segments."
    "Each of the caption may contain following information: \n"
    "1. The main events in the video. \n"
    "2. The main characters in the video. \n"
    "3. The main locations in the video. \n"
    "4. The main objects in the video. \n"
    "5. The main actions in the video. \n"
    "6. The main emotions in the video. \n"
    "Please carefully compare the two captions and merge them into a single caption."
    "Make sure to include all the information from the two captions."
    "Please output the merged caption in the same format as the input captions."
)


class Team(Enum):
    ZUHAO = "zuhao"
    SUDONG = "sudong"
    KAICHEN = "kaichen"
    KEMING = "keming"
    XINGXUAN = "xingxuan"
