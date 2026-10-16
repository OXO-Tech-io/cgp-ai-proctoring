MIN_EYE_OPEN_RATIO = 0.15


def is_eye_open(
    inner_x: float,
    outer_x: float,
    top_y: float,
    bottom_y: float,
) -> bool:
    """Return False when the eyelids are closed (blink) or landmarks are degenerate."""

    width = abs(outer_x - inner_x)
    height = bottom_y - top_y

    return width > 0 and height / width >= MIN_EYE_OPEN_RATIO


class EyeNormalizer:
    """Normalize iris position relative to eye socket boundaries."""

    @staticmethod
    def normalize(
        iris_x: float,
        iris_y: float,
        inner_x: float,
        outer_x: float,
        top_y: float,
        bottom_y: float,
    ) -> tuple[float, float]:
        """Calculate normalized iris position within the eye socket."""

        left_x = min(inner_x, outer_x)
        right_x = max(inner_x, outer_x)

        eye_width = right_x - left_x
        eye_height = bottom_y - top_y

        if eye_width <= 0 or eye_height <= 0:
            raise ValueError("Invalid eye socket boundaries.")

        normalized_x = (iris_x - left_x) / eye_width
        normalized_y = (iris_y - top_y) / eye_height

        return normalized_x, normalized_y

    @staticmethod
    def calculate_offset(
        normalized_x: float,
        normalized_y: float,
    ) -> tuple[float, float]:
        """Calculate iris offset from the center of the eye socket."""

        offset_x = normalized_x - 0.5
        offset_y = normalized_y - 0.5

        return offset_x, offset_y

    @staticmethod
    def is_open(inner_x, outer_x, top_y, bottom_y) -> bool:
        """Alias of is_eye_open."""
        return is_eye_open(inner_x, outer_x, top_y, bottom_y)
