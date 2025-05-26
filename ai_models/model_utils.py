"""
AI模型工具函数
包含模型训练、评估和性能分析功能
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import os
import json
from datetime import datetime
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """训练结果数据类"""
    total_episodes: int
    avg_reward: float
    final_epsilon: float
    total_loss: float
    convergence_episode: Optional[int]
    training_time: float


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    avg_reward: float
    std_reward: float
    win_rate: float
    avg_strategy: float
    strategy_stability: float
    nash_equilibrium_distance: float


class ModelTrainer:
    """
    AI模型训练器
    管理DQN和LSTM模型的训练过程
    """
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.training_history = []
        self.validation_history = []
        
        # 训练参数
        self.save_frequency = config.get('save_frequency', 100)
        self.evaluation_frequency = config.get('evaluation_frequency', 50)
        self.early_stopping_patience = config.get('early_stopping_patience', 200)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        
        # 路径设置
        self.model_save_dir = config.get('model_save_dir', 'results/models')
        self.log_save_dir = config.get('log_save_dir', 'results/logs')
        
        # 确保目录存在
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_save_dir, exist_ok=True)
    
    def train_dqn_agent(self, agent, environment, episodes: int = 1000) -> TrainingResult:
        """
        训练DQN智能体
        
        Args:
            agent: DQN智能体
            environment: 训练环境
            episodes: 训练轮数
            
        Returns:
            训练结果
        """
        start_time = datetime.now()
        episode_rewards = []
        episode_losses = []
        best_avg_reward = float('-inf')
        patience_counter = 0
        convergence_episode = None
        
        logger.info(f"开始训练DQN智能体，计划训练 {episodes} 轮")
        
        for episode in range(episodes):
            # 重置环境
            state = environment.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            
            while not done:
                # 选择动作
                action_result = agent.select_action(state, training=True)
                action = action_result.action
                
                # 执行动作
                next_state, reward, done, info = environment.step(action)
                
                # 存储经验
                agent.store_experience(state, action, reward, next_state, done)
                
                # 训练
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)
                
                state = next_state
                episode_reward += reward
                
                # 更新探索率
                agent.update_exploration_rate(episode, episodes)
            
            episode_rewards.append(episode_reward)
            if episode_loss:
                episode_losses.append(np.mean(episode_loss))
            
            # 记录训练统计
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
                
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, "
                          f"Avg Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.3f}")
                
                # 检查收敛
                if len(episode_rewards) >= 100:
                    recent_rewards = episode_rewards[-100:]
                    reward_std = np.std(recent_rewards)
                    
                    if reward_std < self.convergence_threshold and convergence_episode is None:
                        convergence_episode = episode
                        logger.info(f"模型在第 {episode} 轮收敛")
                
                # 早停检查
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(self.model_save_dir, 'best_dqn_model.pth')
                    agent.save_model(best_model_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"早停触发，在第 {episode} 轮停止训练")
                    break
            
            # 定期保存
            if episode % self.save_frequency == 0 and episode > 0:
                checkpoint_path = os.path.join(self.model_save_dir, f'dqn_checkpoint_{episode}.pth')
                agent.save_model(checkpoint_path)
            
            # 定期评估
            if episode % self.evaluation_frequency == 0 and episode > 0:
                eval_result = self.evaluate_agent(agent, environment, num_episodes=10)
                self.validation_history.append({
                    'episode': episode,
                    'avg_reward': eval_result.avg_reward,
                    'win_rate': eval_result.win_rate,
                    'strategy_stability': eval_result.strategy_stability
                })
        
        # 训练完成
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, 'final_dqn_model.pth')
        agent.save_model(final_model_path)
        
        # 保存训练历史
        self._save_training_history(episode_rewards, episode_losses, 'dqn')
        
        result = TrainingResult(
            total_episodes=len(episode_rewards),
            avg_reward=np.mean(episode_rewards[-100:]),
            final_epsilon=agent.epsilon,
            total_loss=np.mean(episode_losses) if episode_losses else 0,
            convergence_episode=convergence_episode,
            training_time=training_time
        )
        
        logger.info(f"DQN训练完成: {result}")
        return result
    
    def train_lstm_predictor(self, predictor, training_data: List[Dict], 
                           validation_data: List[Dict] = None, epochs: int = 100) -> TrainingResult:
        """
        训练LSTM预测器
        
        Args:
            predictor: LSTM预测器
            training_data: 训练数据
            validation_data: 验证数据
            epochs: 训练轮数
            
        Returns:
            训练结果
        """
        start_time = datetime.now()
        training_losses = []
        validation_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"开始训练LSTM预测器，计划训练 {epochs} 轮")
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_train_losses = []
            
            for batch_data in self._create_lstm_batches(training_data):
                strategies = batch_data['strategies']
                market_states = batch_data['market_states']
                revenues = batch_data['revenues']
                targets = batch_data['targets']
                
                # 训练一批数据
                predictor.train_on_batch(strategies, market_states, revenues, targets)
                
            # 验证阶段
            if validation_data and epoch % 10 == 0:
                val_loss = self._evaluate_lstm_predictor(predictor, validation_data)
                validation_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.6f}")
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(self.model_save_dir, 'best_lstm_model.pth')
                    predictor.save_model(best_model_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience // 10:
                    logger.info(f"LSTM早停触发，在第 {epoch} 轮停止训练")
                    break
        
        # 训练完成
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, 'final_lstm_model.pth')
        predictor.save_model(final_model_path)
        
        result = TrainingResult(
            total_episodes=len(training_losses),
            avg_reward=0,  # LSTM没有reward概念
            final_epsilon=0,
            total_loss=np.mean(validation_losses) if validation_losses else 0,
            convergence_episode=None,
            training_time=training_time
        )
        
        logger.info(f"LSTM训练完成: {result}")
        return result
    
    def _create_lstm_batches(self, data: List[Dict], batch_size: int = 32):
        """创建LSTM训练批次"""
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # 提取批次数据
            batch_strategies = []
            batch_market_states = []
            batch_revenues = []
            batch_targets = []
            
            for item in batch:
                batch_strategies.append(item['strategies'])
                batch_market_states.append(item['market_states'])
                batch_revenues.append(item['revenues'])
                batch_targets.append(item['target_strategy'])
            
            yield {
                'strategies': batch_strategies,
                'market_states': batch_market_states,
                'revenues': batch_revenues,
                'targets': batch_targets
            }
    
    def _evaluate_lstm_predictor(self, predictor, validation_data: List[Dict]) -> float:
        """评估LSTM预测器"""
        total_error = 0
        num_samples = 0
        
        for item in validation_data:
            # 更新历史数据
            for i, (strategy, market_state, revenue) in enumerate(zip(
                item['strategies'], item['market_states'], item['revenues']
            )):
                predictor.update_history(strategy, market_state, revenue)
            
            # 预测
            prediction = predictor.predict_strategy("opponent", 1)
            error = abs(prediction.predicted_strategy - item['target_strategy'])
            total_error += error
            num_samples += 1
        
        return total_error / num_samples if num_samples > 0 else 0
    
    def evaluate_agent(self, agent, environment, num_episodes: int = 100) -> EvaluationResult:
        """
        评估智能体性能
        
        Args:
            agent: 待评估的智能体
            environment: 评估环境
            num_episodes: 评估轮数
            
        Returns:
            评估结果
        """
        episode_rewards = []
        episode_strategies = []
        wins = 0
        
        # 保存原始探索率并设置为评估模式
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # 纯利用模式
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_strategy_list = []
            done = False
            
            while not done:
                action_result = agent.select_action(state, training=False)
                strategy = agent.action_to_strategy(action_result.action)
                episode_strategy_list.append(strategy)
                
                next_state, reward, done, info = environment.step(action_result.action)
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            episode_strategies.extend(episode_strategy_list)
            
            # 判断是否获胜（简化逻辑）
            if episode_reward > 0:
                wins += 1
        
        # 恢复原始探索率
        agent.epsilon = original_epsilon
        
        # 计算统计指标
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        win_rate = wins / num_episodes
        avg_strategy = np.mean(episode_strategies)
        strategy_stability = 1.0 / (1.0 + np.std(episode_strategies))  # 策略稳定性
        
        # 计算纳什均衡距离（简化计算）
        nash_distance = abs(avg_strategy - 30.0) / 20.0  # 假设纳什均衡在30左右
        
        return EvaluationResult(
            avg_reward=avg_reward,
            std_reward=std_reward,
            win_rate=win_rate,
            avg_strategy=avg_strategy,
            strategy_stability=strategy_stability,
            nash_equilibrium_distance=nash_distance
        )
    
    def _save_training_history(self, rewards: List[float], losses: List[float], model_type: str):
        """保存训练历史"""
        history = {
            'rewards': rewards,
            'losses': losses,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type
        }
        
        filename = f'{model_type}_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        filepath = os.path.join(self.log_save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"训练历史已保存到: {filepath}")


class ModelEvaluator:
    """
    模型评估器
    提供详细的模型性能分析和可视化
    """
    
    def __init__(self, config: Dict):
        """初始化评估器"""
        self.config = config
        self.results_dir = config.get('results_dir', 'results/plots')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def comprehensive_evaluation(self, agents: List[Any], environment, 
                               num_episodes: int = 200) -> Dict[str, Any]:
        """
        综合评估多个智能体
        
        Args:
            agents: 智能体列表
            environment: 评估环境
            num_episodes: 评估轮数
            
        Returns:
            详细评估结果
        """
        results = {}
        
        for i, agent in enumerate(agents):
            agent_name = f"Agent_{i}"
            logger.info(f"评估 {agent_name}...")
            
            # 基础性能评估
            basic_eval = self._basic_performance_evaluation(agent, environment, num_episodes)
            
            # 策略分析
            strategy_analysis = self._strategy_analysis(agent, environment, num_episodes)
            
            # 适应性分析
            adaptability = self._adaptability_analysis(agent, environment)
            
            results[agent_name] = {
                'basic_performance': basic_eval,
                'strategy_analysis': strategy_analysis,
                'adaptability': adaptability
            }
        
        # 比较分析
        comparison = self._comparative_analysis(results)
        results['comparison'] = comparison
        
        # 生成报告
        self._generate_evaluation_report(results)
        
        return results
    
    def _basic_performance_evaluation(self, agent, environment, num_episodes: int) -> Dict:
        """基础性能评估"""
        episode_rewards = []
        episode_lengths = []
        strategy_sequences = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_length = 0
            strategy_sequence = []
            done = False
            
            while not done:
                action_result = agent.select_action(state, training=False)
                strategy = agent.action_to_strategy(action_result.action)
                strategy_sequence.append(strategy)
                
                next_state, reward, done, info = environment.step(action_result.action)
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            strategy_sequences.append(strategy_sequence)
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'reward_trend': self._calculate_trend(episode_rewards),
            'strategy_diversity': self._calculate_strategy_diversity(strategy_sequences)
        }
    
    def _strategy_analysis(self, agent, environment, num_episodes: int) -> Dict:
        """策略分析"""
        all_strategies = []
        strategy_changes = []
        decision_confidence = []
        
        for episode in range(min(num_episodes, 50)):  # 限制分析轮数
            state = environment.reset()
            episode_strategies = []
            episode_confidence = []
            done = False
            
            while not done:
                action_result = agent.select_action(state, training=False)
                strategy = agent.action_to_strategy(action_result.action)
                episode_strategies.append(strategy)
                episode_confidence.append(action_result.confidence)
                
                next_state, reward, done, info = environment.step(action_result.action)
                state = next_state
            
            all_strategies.extend(episode_strategies)
            decision_confidence.extend(episode_confidence)
            
            # 计算策略变化
            if len(episode_strategies) > 1:
                changes = [abs(episode_strategies[i] - episode_strategies[i-1]) 
                          for i in range(1, len(episode_strategies))]
                strategy_changes.extend(changes)
        
        return {
            'avg_strategy': np.mean(all_strategies),
            'strategy_std': np.std(all_strategies),
            'strategy_range': (np.min(all_strategies), np.max(all_strategies)),
            'avg_strategy_change': np.mean(strategy_changes) if strategy_changes else 0,
            'avg_decision_confidence': np.mean(decision_confidence),
            'strategy_stability': 1.0 / (1.0 + np.std(all_strategies))
        }
    
    def _adaptability_analysis(self, agent, environment) -> Dict:
        """适应性分析"""
        # 测试不同市场条件下的表现
        test_conditions = [
            {'market_volatility': 'low', 'competition': 'low'},
            {'market_volatility': 'high', 'competition': 'low'},
            {'market_volatility': 'low', 'competition': 'high'},
            {'market_volatility': 'high', 'competition': 'high'}
        ]
        
        adaptability_scores = {}
        
        for condition in test_conditions:
            # 设置测试环境（这里简化处理）
            test_rewards = []
            
            for _ in range(20):  # 每个条件测试20轮
                state = environment.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action_result = agent.select_action(state, training=False)
                    next_state, reward, done, info = environment.step(action_result.action)
                    state = next_state
                    episode_reward += reward
                
                test_rewards.append(episode_reward)
            
            condition_key = f"{condition['market_volatility']}_{condition['competition']}"
            adaptability_scores[condition_key] = np.mean(test_rewards)
        
        return adaptability_scores
    
    def _comparative_analysis(self, results: Dict) -> Dict:
        """比较分析"""
        agents = [k for k in results.keys() if k != 'comparison']
        
        # 性能排名
        performance_ranking = sorted(agents, 
                                   key=lambda x: results[x]['basic_performance']['avg_reward'], 
                                   reverse=True)
        
        # 策略多样性比较
        strategy_diversity = {agent: results[agent]['strategy_analysis']['strategy_std'] 
                            for agent in agents}
        
        # 稳定性比较
        stability_ranking = sorted(agents,
                                 key=lambda x: results[x]['strategy_analysis']['strategy_stability'],
                                 reverse=True)
        
        return {
            'performance_ranking': performance_ranking,
            'strategy_diversity': strategy_diversity,
            'stability_ranking': stability_ranking,
            'best_overall': performance_ranking[0] if performance_ranking else None
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单线性回归
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_strategy_diversity(self, strategy_sequences: List[List[float]]) -> float:
        """计算策略多样性"""
        all_strategies = []
        for seq in strategy_sequences:
            all_strategies.extend(seq)
        
        if not all_strategies:
            return 0.0
        
        # 使用标准差作为多样性指标
        return np.std(all_strategies)
    
    def plot_training_progress(self, training_history: Dict, save_path: str = None):
        """绘制训练进度图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        rewards = training_history['rewards']
        ax1.plot(rewards, alpha=0.3, color='blue')
        # 移动平均
        window = min(50, len(rewards)//10)
        if window > 1:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            ax1.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        if 'losses' in training_history and training_history['losses']:
            losses = training_history['losses']
            ax2.plot(losses, color='orange', alpha=0.7)
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # 奖励分布
        ax3.hist(rewards, bins=30, alpha=0.7, color='green')
        ax3.set_title('Reward Distribution')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # 性能趋势
        episode_windows = [rewards[i:i+100] for i in range(0, len(rewards)-100, 50)]
        avg_rewards = [np.mean(window) for window in episode_windows]
        ax4.plot(range(0, len(rewards)-100, 50), avg_rewards, marker='o')
        ax4.set_title('Performance Trend (100-episode windows)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练进度图已保存到: {save_path}")
        
        return fig
    
    def _generate_evaluation_report(self, results: Dict):
        """生成评估报告"""
        report_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AI模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 比较结果
            if 'comparison' in results:
                comp = results['comparison']
                f.write("性能排名:\n")
                for i, agent in enumerate(comp['performance_ranking'], 1):
                    f.write(f"  {i}. {agent}\n")
                f.write(f"\n最佳智能体: {comp['best_overall']}\n\n")
            
            # 详细结果
            for agent_name, agent_results in results.items():
                if agent_name == 'comparison':
                    continue
                
                f.write(f"{agent_name} 详细评估结果:\n")
                f.write("-" * 30 + "\n")
                
                basic = agent_results['basic_performance']
                f.write(f"  平均奖励: {basic['avg_reward']:.3f}\n")
                f.write(f"  奖励标准差: {basic['std_reward']:.3f}\n")
                f.write(f"  最大奖励: {basic['max_reward']:.3f}\n")
                f.write(f"  最小奖励: {basic['min_reward']:.3f}\n")
                
                strategy = agent_results['strategy_analysis']
                f.write(f"  平均策略: {strategy['avg_strategy']:.3f}\n")
                f.write(f"  策略稳定性: {strategy['strategy_stability']:.3f}\n")
                f.write(f"  决策置信度: {strategy['avg_decision_confidence']:.3f}\n")
                
                f.write("\n")
        
        logger.info(f"评估报告已生成: {report_path}") 