from simulators.example_joystick import JoystickRandom


def test_joystick():
    J = JoystickRandom()
    J.establish_sender_connection()
    J.establish_receiver_connection()
    # first listen() for the episode names
    assert(J.get_all_episode_names())
    episodes = J.get_episodes()
    # we want to run on at least one episode
    assert(len(episodes) > 0)
    for ep_title in episodes:
        print("Waiting for episode: {}".format(ep_title))
        # second listen() for the specific episode details
        J.get_episode_metadata()
        assert(J.current_ep and J.current_ep.get_name() == ep_title)
        J.init_control_pipeline()
        J.update_loop()


if __name__ == '__main__':
    print("Joystick")
    test_joystick()
