radiusPourcentageTable = {'5fr': 1,
                          '2fr': 0.87122,
                          '1fr': 0.73767,
                          '20c': 0.66931,
                          '10c': 0.60890,
                          '50c': 0.57869,
                          '5c': 0.54531}


def predict_piece_by_radius_pourcentage(radius):
    keys = list(radiusPourcentageTable.keys())
    values = list(radiusPourcentageTable.values())
    last = False

    if radius >= values[0]:
        return keys[0]

    for index in range(len(keys) - 1):
        top_slice_key = keys[index]
        top_slice_value = values[index]

        bot_slice_key = keys[index + 1]
        bot_slice_value = values[index + 1]

        diff_top = abs(top_slice_value - radius)
        diff_bot = abs(radius - bot_slice_value)

        if radius <= top_slice_value and radius >= bot_slice_value:
            if diff_top < diff_bot:
                return top_slice_key
            else:
                return bot_slice_key
    return keys[-1]


if __name__ == '__main__':
    print(predict_piece_by_radius_pourcentage(1.2))
