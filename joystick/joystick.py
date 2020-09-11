from joystick.example_joystick import Joystick


def test_joystick():
    J = Joystick()
    J.establish_sender_connection()
    J.establish_receiver_connection()
    # TODO: return an int encoding the manage_data type in listen_once()
    assert(J.listen_once())  # first listen() for the episode names
    episodes = J.get_episodes()
    # we want to run on at least one episode
    assert(len(episodes) > 0)
    for ep_title in episodes:
        print("Waiting for episode: {}".format(ep_title))
        J.listen_once()  # second listen() for the specific episode details
        assert(J.current_ep and J.current_ep.get_name() == ep_title)
        J.init_control_pipeline()
        J.update()
    print("Finished joystick test")


if __name__ == '__main__':
    print("Joystick")
    test_joystick()
