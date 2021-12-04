from phue import Bridge

if __name__ == '__main__':

    b = Bridge( '192.168.1.101' )

    b.connect()

    for light in b.lights:
        if not light.reachable:
            continue

        light.on = not light.on
