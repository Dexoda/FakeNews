import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .charts import create_chart
from .heatmap import create_text_heatmap

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Генератор отчетов по результатам анализа новостей.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализирует генератор отчетов с настройками из конфигурации.

        Args:
            config: Настройки визуализации из конфигурационного файла
        """
        self.config = config or {}

    def generate_report(
        self,
        text: str,
        analysis_results: Dict[str, Any],
        factcheck_results: List[Dict[str, Any]],
        sources_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Создает полный отчет на основе результатов анализа.

        Args:
            text: Исходный текст новости
            analysis_results: Результаты анализа текста
            factcheck_results: Результаты проверки фактов
            sources_results: Результаты поиска по открытым источникам

        Returns:
            Структурированный отчет с визуализациями
        """
        try:
            # Формируем итоговую оценку достоверности
            credibility_score = self.calculate_total_credibility_score(
                analysis_results, factcheck_results, sources_results
            )

            # Собираем все подозрительные фрагменты
            suspicious_fragments = self.collect_suspicious_fragments(analysis_results)

            # Создаем тепловую карту текста
            text_heatmap = create_text_heatmap(text, suspicious_fragments, 'html', self.config)

            # Создаем графики
            charts = self.generate_charts(analysis_results, factcheck_results, sources_results, credibility_score)

            # Формируем заключение и рекомендации
            conclusion, recommendations = self.generate_conclusion_and_recommendations(
                credibility_score, analysis_results, factcheck_results, sources_results
            )

            # Создаем структуру отчета
            report = {
                "text": text,
                "credibility_score": credibility_score,
                "credibility_level": self.get_credibility_level(credibility_score),
                "text_heatmap": text_heatmap,
                "charts": charts,
                "analysis_summary": self.generate_analysis_summary(analysis_results),
                "factcheck_summary": self.generate_factcheck_summary(factcheck_results),
                "sources_summary": self.generate_sources_summary(sources_results),
                "suspicious_fragments": suspicious_fragments,
                "conclusion": conclusion,
                "recommendations": recommendations,
                "generation_time": datetime.now().isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"Ошибка при создании отчета: {e}")

            # Возвращаем базовый отчет с ошибкой
            return {
                "text": text,
                "error": str(e),
                "credibility_score": 0.5,
                "credibility_level": "не определено",
                "conclusion": "Произошла ошибка при создании отчета."
            }

    def calculate_total_credibility_score(
        self,
        analysis_results: Dict[str, Any],
        factcheck_results: List[Dict[str, Any]],
        sources_results: List[Dict[str, Any]]
    ) -> float:
        """
        Рассчитывает итоговую оценку достоверности на основе всех результатов.

        Args:
            analysis_results: Результаты анализа текста
            factcheck_results: Результаты проверки фактов
            sources_results: Результаты поиска по открытым источникам

        Returns:
            Оценка достоверности от 0 до 1
        """
        # Весовые коэффициенты для разных компонентов
        weights = {
            'statistical': 0.1,
            'linguistic': 0.2,
            'semantic': 0.2,
            'structural': 0.1,
            'factcheck': 0.2,
            'sources': 0.2
        }

        # Извлекаем оценки из результатов анализа
        scores = {}

        # Оценки из лингвистического анализа
        if 'statistical' in analysis_results:
            scores['statistical'] = analysis_results['statistical'].get('credibility_score', 0.5)

        if 'linguistic' in analysis_results:
            scores['linguistic'] = analysis_results['linguistic'].get('credibility_score', 0.5)

        if 'semantic' in analysis_results:
            scores['semantic'] = analysis_results['semantic'].get('credibility_score', 0.5)

        if 'structural' in analysis_results:
            scores['structural'] = analysis_results['structural'].get('credibility_score', 0.5)

        # Оценка из проверки фактов
        if factcheck_results:
            factcheck_scores = []
            for result in factcheck_results:
                factcheck_scores.append(result.get('rating', 0.5))

            if factcheck_scores:
                scores['factcheck'] = sum(factcheck_scores) / len(factcheck_scores)
            else:
                scores['factcheck'] = 0.5

        # Оценка из поиска по открытым источникам
        if sources_results:
            source_scores = []
            for result in sources_results:
                source_scores.append(result.get('confirmation_score', 0.5))

            if source_scores:
                scores['sources'] = sum(source_scores) / len(source_scores)
            else:
                scores['sources'] = 0.5

        # Вычисляем взвешенную сумму
        total_score = 0
        total_weight = 0

        for component, score in scores.items():
            weight = weights.get(component, 0)
            total_score += score * weight
            total_weight += weight

        # Если нет оценок, возвращаем нейтральную оценку
        if total_weight == 0:
            return 0.5

        # Нормализуем итоговую оценку
        return total_score / total_weight

    def collect_suspicious_fragments(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Собирает все подозрительные фрагменты из разных типов анализа.

        Args:
            analysis_results: Результаты анализа текста

        Returns:
            Список всех подозрительных фрагментов
        """
        suspicious_fragments = []

        # Собираем фрагменты из разных модулей анализа
        if 'linguistic' in analysis_results:
            fragments = analysis_results['linguistic'].get('suspicious_fragments', [])
            for fragment in fragments:
                suspicious_fragments.append(fragment)

        if 'semantic' in analysis_results:
            fragments = analysis_results['semantic'].get('suspicious_fragments', [])
            for fragment in fragments:
                suspicious_fragments.append(fragment)

        # Удаляем дубликаты (фрагменты с одинаковыми start/end)
        unique_fragments = {}
        for fragment in suspicious_fragments:
            key = f"{fragment.get('start', 0)}_{fragment.get('end', 0)}"

            if key not in unique_fragments or fragment.get('confidence', 0) > unique_fragments[key].get('confidence', 0):
                unique_fragments[key] = fragment

        return list(unique_fragments.values())

    def generate_charts(
        self,
        analysis_results: Dict[str, Any],
        factcheck_results: List[Dict[str, Any]],
        sources_results: List[Dict[str, Any]],
        credibility_score: float
    ) -> Dict[str, str]:
        """
        Создает все графики для отчета.

        Args:
            analysis_results: Результаты анализа текста
            factcheck_results: Результаты проверки фактов
            sources_results: Результаты поиска по открытым источникам
            credibility_score: Итоговая оценка достоверности

        Returns:
            Словарь с графиками в формате base64
        """
        charts = {}

        # Создаем индикатор общей оценки достоверности
        charts['credibility_gauge'] = create_chart('credibility_gauge', {
            'score': credibility_score,
            'title': 'Общая оценка достоверности'
        }, self.config)

        # Создаем диаграмму соотношения фактов и мнений
        if 'statistical' in analysis_results:
            charts['fact_opinion_pie'] = create_chart('fact_opinion_pie', {
                'fact_ratio': analysis_results['statistical'].get('fact_opinion_ratio', 0.5)
            }, self.config)

        # Создаем столбчатую диаграмму для оценок разных типов анализа
        credibility_scores = {}

        if 'statistical' in analysis_results:
            credibility_scores['Статистический'] = analysis_results['statistical'].get('credibility_score', 0.5)

        if 'linguistic' in analysis_results:
            credibility_scores['Лингвистический'] = analysis_results['linguistic'].get('credibility_score', 0.5)

        if 'semantic' in analysis_results:
            credibility_scores['Семантический'] = analysis_results['semantic'].get('credibility_score', 0.5)

        if 'structural' in analysis_results:
            credibility_scores['Структурный'] = analysis_results['structural'].get('credibility_score', 0.5)

        if credibility_scores:
            charts['analysis_types_bar'] = create_chart('analysis_types_bar', {
                'credibility_scores': credibility_scores
            }, self.config)

        # Создаем диаграмму результатов проверки фактов
        if factcheck_results:
            charts['factcheck_bar'] = create_chart('factcheck_bar', {
                'factcheck_results': factcheck_results
            }, self.config)

        # Создаем диаграмму эмоциональных маркеров
        if 'linguistic' in analysis_results and 'emotional_markers' in analysis_results['linguistic']:
            charts['emotion_bar'] = create_chart('emotion_bar', {
                'emotional_markers': analysis_results['linguistic']['emotional_markers']
            }, self.config)

        # Создаем диаграмму манипулятивных конструкций
        if 'linguistic' in analysis_results and 'manipulative_constructs' in analysis_results['linguistic']:
            charts['manipulative_constructs_chart'] = create_chart('manipulative_constructs_chart', {
                'constructs': analysis_results['linguistic']['manipulative_constructs']
            }, self.config)

        return charts

    def generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создает сводку результатов анализа текста.

        Args:
            analysis_results: Результаты анализа текста

        Returns:
            Сводка по каждому типу анализа
        """
        summary = {}

        # Сводка статистического анализа
        if 'statistical' in analysis_results:
            statistical = analysis_results['statistical']
            summary['statistical'] = {
                'word_count': statistical.get('word_count', 0),
                'sentence_count': statistical.get('sentence_count', 0),
                'avg_sentence_length': statistical.get('avg_sentence_length', 0),
                'readability_score': statistical.get('readability_score', 0),
                'fact_opinion_ratio': statistical.get('fact_opinion_ratio', 0),
                'credibility_score': statistical.get('credibility_score', 0)
            }

        # Сводка лингвистического анализа
        if 'linguistic' in analysis_results:
            linguistic = analysis_results['linguistic']
            summary['linguistic'] = {
                'sentiment': linguistic.get('sentiment', 'не определено'),
                'emotional_tone': linguistic.get('emotional_tone', 'не определено'),
                'manipulative_constructs_count': linguistic.get('manipulative_constructs_count', 0),
                'credibility_score': linguistic.get('credibility_score', 0)
            }

        # Сводка семантического анализа
        if 'semantic' in analysis_results:
            semantic = analysis_results['semantic']
            summary['semantic'] = {
                'key_themes': semantic.get('key_themes', [])[:5],  # Ограничиваем до 5 тем
                'coherence': semantic.get('coherence', {}).get('coherence_score', 0),
                'logical_flow': semantic.get('coherence', {}).get('logical_flow', 'не определено'),
                'contradictions_count': semantic.get('contradictions_count', 0),
                'credibility_score': semantic.get('credibility_score', 0)
            }

        # Сводка структурного анализа
        if 'structural' in analysis_results:
            structural = analysis_results['structural']
            summary['structural'] = {
                'structure_quality': structural.get('structure_quality', 'не определено'),
                'journalism_standards_score': structural.get('journalism_standards_score', 0),
                'sources_analysis': {
                    'total_sources': structural.get('sources_analysis', {}).get('total_sources', 0),
                    'reliability_score': structural.get('sources_analysis', {}).get('reliability_score', 0)
                },
                'credibility_score': structural.get('credibility_score', 0)
            }

        return summary

    def generate_factcheck_summary(self, factcheck_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Создает сводку результатов проверки фактов.

        Args:
            factcheck_results: Результаты проверки фактов

        Returns:
            Сводка по проверке фактов
        """
        if not factcheck_results:
            return {
                'checked_claims_count': 0,
                'status': 'Проверка не выполнена'
            }

        # Подсчитываем количество утверждений по статусам
        status_counts = {
            'подтверждено': 0,
            'вероятно правда': 0,
            'спорно': 0,
            'вероятно ложь': 0,
            'опровергнуто': 0,
            'не проверено': 0
        }

        for result in factcheck_results:
            status = result.get('status', 'не проверено')
            status_counts[status] = status_counts.get(status, 0) + 1

        # Определяем общий статус
        if status_counts['опровергнуто'] > 0:
            general_status = 'Обнаружены опровергнутые утверждения'
        elif status_counts['вероятно ложь'] > 0:
            general_status = 'Обнаружены сомнительные утверждения'
        elif status_counts['подтверждено'] > 0 or status_counts['вероятно правда'] > 0:
            general_status = 'Обнаружены подтвержденные утверждения'
        else:
            general_status = 'Статус утверждений не определен'

        # Создаем сводку
        summary = {
            'checked_claims_count': len(factcheck_results),
            'status': general_status,
            'status_counts': status_counts,
            'verified_claims': []
        }

        # Добавляем первые 5 проверенных утверждений для сводки
        for result in factcheck_results[:5]:
            if result.get('status') != 'не проверено':
                summary['verified_claims'].append({
                    'claim': result.get('claim', ''),
                    'status': result.get('status', 'не проверено'),
                    'sources_count': len(result.get('sources', []))
                })

        return summary

    def generate_sources_summary(self, sources_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Создает сводку результатов поиска по открытым источникам.

        Args:
            sources_results: Результаты поиска по открытым источникам

        Returns:
            Сводка по поиску в открытых источниках
        """
        if not sources_results:
            return {
                'checked_claims_count': 0,
                'status': 'Поиск не выполнен'
            }

        # Подсчитываем количество утверждений по уровням подтверждения
        confirmation_counts = {
            'подтверждено': 0,
            'вероятно правда': 0,
            'неопределенно': 0,
            'вероятно ложь': 0,
            'опровергнуто': 0
        }

        total_sources = 0
        credible_sources = 0

        for result in sources_results:
            confirmation_level = result.get('confirmation_level', 'неопределенно')
            confirmation_counts[confirmation_level] = confirmation_counts.get(confirmation_level, 0) + 1

            # Подсчитываем общее количество источников
            total_sources += len(result.get('sources', []))
            credible_sources += result.get('credible_sources_count', 0)

        # Определяем общий статус
        if confirmation_counts['опровергнуто'] > 0:
            general_status = 'Обнаружены опровергнутые утверждения'
        elif confirmation_counts['вероятно ложь'] > 0:
            general_status = 'Обнаружены сомнительные утверждения'
        elif confirmation_counts['подтверждено'] > 0 or confirmation_counts['вероятно правда'] > 0:
            general_status = 'Обнаружены подтвержденные утверждения'
        else:
            general_status = 'Статус утверждений не определен'

        # Создаем сводку
        summary = {
            'checked_claims_count': len(sources_results),
            'status': general_status,
            'confirmation_counts': confirmation_counts,
            'total_sources_found': total_sources,
            'credible_sources_count': credible_sources,
            'verified_claims': []
        }

        # Добавляем первые 5 проверенных утверждений для сводки
        for result in sources_results[:5]:
            if result.get('confirmation_level') != 'неопределенно':
                summary['verified_claims'].append({
                    'claim': result.get('claim', ''),
                    'confirmation_level': result.get('confirmation_level', 'неопределенно'),
                    'supporting_sources_count': len(result.get('supporting_sources', [])),
                    'contradicting_sources_count': len(result.get('contradicting_sources', []))
                })

        return summary

    def generate_conclusion_and_recommendations(
        self,
        credibility_score: float,
        analysis_results: Dict[str, Any],
        factcheck_results: List[Dict[str, Any]],
        sources_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Формирует заключение и рекомендации на основе результатов анализа.

        Args:
            credibility_score: Итоговая оценка достоверности
            analysis_results: Результаты анализа текста
            factcheck_results: Результаты проверки фактов
            sources_results: Результаты поиска по открытым источникам

        Returns:
            Кортеж из заключения и списка рекомендаций
        """
        # Формируем заключение в зависимости от оценки достоверности
        conclusion = ""
        if credibility_score >= 0.8:
            conclusion = "Текст имеет высокую степень достоверности. Анализ не выявил существенных признаков недостоверной информации."
        elif credibility_score >= 0.6:
            conclusion = "Текст имеет среднюю степень достоверности. Обнаружены некоторые признаки, требующие дополнительной проверки."
        elif credibility_score >= 0.4:
            conclusion = "Текст имеет низкую степень достоверности. Обнаружены многочисленные признаки манипуляций и/или недостоверной информации."
        else:
            conclusion = "Текст, вероятно, содержит недостоверную информацию. Обнаружены явные признаки фейковых новостей."

        # Дополняем заключение деталями из разных модулей анализа
        details = []

        # Статистический анализ
        if 'statistical' in analysis_results:
            fact_opinion_ratio = analysis_results['statistical'].get('fact_opinion_ratio', 0.5)
            if fact_opinion_ratio < 0.3:
                details.append("Текст содержит преимущественно мнения и субъективные оценки, а не факты.")
            elif fact_opinion_ratio > 0.7:
                details.append("Текст содержит преимущественно фактическую информацию.")

        # Лингвистический анализ
        if 'linguistic' in analysis_results:
            manipulative_count = analysis_results['linguistic'].get('manipulative_constructs_count', 0)
            if manipulative_count > 5:
                details.append(f"Обнаружено значительное количество манипулятивных конструкций ({manipulative_count}).")

            emotional_tone = analysis_results['linguistic'].get('emotional_tone', '')
            if 'негативный' in emotional_tone or 'преувеличение' in emotional_tone:
                details.append(f"Текст имеет выраженную эмоциональную окраску ({emotional_tone}).")

        # Семантический анализ
        if 'semantic' in analysis_results:
            contradictions_count = analysis_results['semantic'].get('contradictions_count', 0)
            if contradictions_count > 0:
                details.append(f"В тексте обнаружены противоречия ({contradictions_count}).")

            logical_flow = analysis_results['semantic'].get('coherence', {}).get('logical_flow', '')
            if logical_flow in ['слабый', 'нарушенный']:
                details.append(f"Текст имеет нарушения логической связности ({logical_flow}).")

        # Структурный анализ
        if 'structural' in analysis_results:
            structure_quality = analysis_results['structural'].get('structure_quality', '')
            if structure_quality in ['слабая', 'плохая']:
                details.append(f"Структура текста не соответствует стандартам качественной журналистики ({structure_quality}).")

            total_sources = analysis_results['structural'].get('sources_analysis', {}).get('total_sources', 0)
            if total_sources == 0:
                details.append("В тексте отсутствуют ссылки на источники информации.")

        # Проверка фактов
        if factcheck_results:
            false_claims = sum(1 for r in factcheck_results
                            if r.get('status') in ['опровергнуто', 'вероятно ложь'])
            if false_claims > 0:
                details.append(f"Обнаружены опровергнутые или сомнительные утверждения ({false_claims}).")

        # Поиск по открытым источникам
        if sources_results:
            contradicting_claims = sum(1 for r in sources_results
                                    if r.get('confirmation_level') in ['опровергнуто', 'вероятно ложь'])
            if contradicting_claims > 0:
                details.append(f"Информация в тексте противоречит данным из открытых источников ({contradicting_claims} утверждений).")

        # Добавляем детали к заключению
        if details:
            conclusion += " " + " ".join(details)

        # Формируем рекомендации
        recommendations = []

        if credibility_score < 0.6:
            recommendations.append("Проверьте информацию в других авторитетных источниках.")

        if 'statistical' in analysis_results and analysis_results['statistical'].get('fact_opinion_ratio', 0.5) < 0.4:
            recommendations.append("Обратите внимание на соотношение фактов и мнений в тексте.")

        if 'linguistic' in analysis_results and analysis_results['linguistic'].get('manipulative_constructs_count', 0) > 3:
            recommendations.append("Критически оценивайте манипулятивные речевые конструкции в тексте.")

        if 'semantic' in analysis_results and analysis_results['semantic'].get('contradictions_count', 0) > 0:
            recommendations.append("Обратите внимание на противоречия в тексте.")

        if 'structural' in analysis_results and analysis_results['structural'].get('sources_analysis', {}).get('total_sources', 0) == 0:
            recommendations.append("Ищите тексты с указанием источников информации.")

        # Если рекомендаций мало, добавляем общие
        if len(recommendations) < 2:
            recommendations.append("Сравнивайте информацию с другими надежными источниками.")
            recommendations.append("Обращайте внимание на эмоциональную окраску и манипулятивные приемы в тексте.")

        return conclusion, recommendations

    def get_credibility_level(self, score: float) -> str:
        """
        Определяет уровень достоверности на основе числовой оценки.

        Args:
            score: Оценка достоверности от 0 до 1

        Returns:
            Текстовое описание уровня достоверности
        """
        if score >= 0.8:
            return "высокая достоверность"
        elif score >= 0.6:
            return "средняя достоверность"
        elif score >= 0.4:
            return "низкая достоверность"
        else:
            return "вероятный фейк"

async def generate_report(
    text: str,
    analysis_results: Dict[str, Any],
    factcheck_results: List[Dict[str, Any]],
    sources_results: List[Dict[str, Any]],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Создает полный отчет на основе результатов анализа.

    Args:
        text: Исходный текст новости
        analysis_results: Результаты анализа текста
        factcheck_results: Результаты проверки фактов
        sources_results: Результаты поиска по открытым источникам
        config: Дополнительные настройки

    Returns:
        Структурированный отчет с визуализациями
    """
    generator = ReportGenerator(config)
    return generator.generate_report(text, analysis_results, factcheck_results, sources_results)
