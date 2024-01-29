import gymnasium as gym
from aitool import make_dir
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    make_dir('./recorder')
    video_recorder = VideoRecorder(env, path='./recorder/tmp.mp4')

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        video_recorder.capture_frame()
        obs, reward, done, _, _ = env.step(action)
        print(obs)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    video_recorder.close()
    video_recorder.enabled = False
    env.close()
