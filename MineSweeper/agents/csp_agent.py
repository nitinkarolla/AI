class CSPAgent():

    def __init__(self, env = None):
        self.env = env

    def play(self):
        self.env.click_square(0, 0)
        self.env.render_env()
        return
