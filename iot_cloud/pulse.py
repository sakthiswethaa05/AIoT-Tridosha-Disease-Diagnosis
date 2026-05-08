import serial

def get_pulse_output():

    ser = serial.Serial('COM3', 115200)

    pulse = ""

    while True:
        try:
            data = ser.readline().decode().strip()

            if "PULSE:" in data:
                pulse = data.split(":")[1]
                break

        except:
            continue

    ser.close()

    return pulse