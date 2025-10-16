#!/usr/bin/env python3
"""
Aplicação de cálculo e simulação de condutas AVAC
-------------------------------------------------

Esta aplicação fornece duas funcionalidades principais:

1. **Seleção de condutas (modo `select`)**: a partir de um caudal conhecido
   (m³/h) e de um critério de dimensionamento (velocidade do ar ou perda de
   carga por metro), calcula automaticamente as dimensões da conduta.
   - Para condutas **circulares**, calcula o diâmetro.
   - Para condutas **retangulares**, calcula as dimensões dos lados a e b
     com base no rácio de forma (aspect ratio).
   - Devolve ainda a velocidade, perda de carga por metro, número de Reynolds
     e outras grandezas úteis.

2. **Simulação de rede de condutas (modo `network`)**: a partir de um
   caudal, dimensão inicial (diâmetro ou lados) e de uma lista de segmentos
   (reta, curva, redução), calcula as perdas de carga distribuídas e locais
   ao longo de todo o percurso. O resultado é gravado num relatório HTML
   (`relatorio.html`) com um desenho SVG da rede (`rede.svg`) incorporado.

3. **Testes automáticos (modo `--test`)**: validações básicas das funções de
   dimensionamento e cálculo. Quando executado com `python app.py --test`,
   corre uma bateria de testes e indica se todos passaram.

Ao conceber esta aplicação foram evitadas dependências externas como
`tkinter` ou `matplotlib`, para garantir compatibilidade com ambientes
sandbox (por exemplo, nos módulos ChatGPT). O desenho da rede é gerado em
SVG puro.

Uso simplificado (exemplos):

```
# Selecionar conduta circular a partir do caudal e da velocidade
python app.py select --shape circular --flow 1800 --vel 5

# Selecionar conduta circular a partir do caudal e da perda de carga
python app.py select --shape circular --flow 1800 --dp_per_m 0.9

# Selecionar conduta retangular (rácio 2:1) a partir do caudal e da velocidade
python app.py select --shape rectangular --flow 2500 --vel 5 --aspect 2

# Simular rede de condutas circulares
python app.py network \
    --shape circular \
    --flow 3000 \
    --diameter 0.315 \
    --segments '[{"type":"straight","L":12},{"type":"elbow90"},{"type":"reducer","D2":0.25},{"type":"straight","L":5}]'

# Executar testes unitários
python app.py --test
```

Autor: ChatGPT
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple, Dict, Any


# Constantes físicas (ar a 20°C aproximadamente)
RHO_AIR = 1.2        # densidade do ar (kg/m³)
MU_AIR = 1.8e-5       # viscosidade dinâmica (Pa·s)

# Factor de atrito típico para condutas metálicas lisas. Em projectos reais
# este valor depende do número de Reynolds e da rugosidade relativa, mas aqui
# assume-se um valor constante para simplificação.
FRICTION_FACTOR = 0.02

def compute_reynolds(rho: float, velocity: float, diameter: float) -> float:
    """Calcula o número de Reynolds para escoamento em conduta circular.
    Para condutas retangulares utiliza-se o diâmetro equivalente.

    Re = (ρ · V · D) / μ
    """
    if diameter <= 0:
        return float('nan')
    return rho * velocity * diameter / MU_AIR

def friction_pressure_drop(f: float, rho: float, velocity: float, diameter: float) -> float:
    """Calcula a perda de carga distribuída por metro (Pa/m) para uma conduta.

    Fórmula de Darcy-Weisbach simplificada:

        Δp/m = f · (ρ·V²)/(2·D)

    """
    if diameter <= 0:
        return float('nan')
    return f * (rho * velocity * velocity) / (2.0 * diameter)

def compute_circular_from_velocity(flow: float, velocity: float) -> Dict[str, float]:
    """Dimensiona uma conduta circular a partir do caudal e da velocidade.

    :param flow: Caudal (m³/h)
    :param velocity: Velocidade (m/s)
    :return: dicionário com diâmetro (m), área (m²), velocidade (m/s) calculada,
             perda de carga por metro (Pa/m) e Re.
    """
    if velocity <= 0:
        raise ValueError("A velocidade deve ser positiva.")
    # Conversão para m³/s
    q_m = flow / 3600.0
    # Área transversal necessária (m²)
    area = q_m / velocity
    # Diâmetro correspondente (m)
    diameter = math.sqrt(4.0 * area / math.pi)
    # Perda de carga distribuída por metro
    dp_per_m = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity, diameter)
    # Número de Reynolds
    Re = compute_reynolds(RHO_AIR, velocity, diameter)
    return {
        'diameter': diameter,
        'area': area,
        'velocity': velocity,
        'dp_per_m': dp_per_m,
        'Re': Re
    }

def compute_circular_from_dp(flow: float, dp_per_m_target: float) -> Dict[str, float]:
    """Dimensiona uma conduta circular a partir do caudal e da perda de carga alvo.

    Resolve D^5 = (f·ρ/2 · Q²) / ((π/4)² · Δp)
    """
    if dp_per_m_target <= 0:
        raise ValueError("A perda de carga deve ser positiva.")
    q_m = flow / 3600.0
    numerator = FRICTION_FACTOR * (RHO_AIR / 2.0) * (q_m ** 2)
    denominator = ((math.pi / 4.0) ** 2) * dp_per_m_target
    d_power5 = numerator / denominator
    diameter = d_power5 ** 0.2  # 1/5
    area = math.pi * (diameter ** 2) / 4.0
    velocity = q_m / area
    dp_per_m_calc = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity, diameter)
    Re = compute_reynolds(RHO_AIR, velocity, diameter)
    return {
        'diameter': diameter,
        'area': area,
        'velocity': velocity,
        'dp_per_m': dp_per_m_calc,
        'Re': Re
    }

def compute_rectangular_from_velocity(flow: float, velocity: float, aspect: float) -> Dict[str, float]:
    """Dimensiona uma conduta retangular a partir do caudal, velocidade e rácio de forma (a/b).

    O utilizador deve fornecer o rácio `aspect` = a/b ≥ 1. O código calcula a secção
    transversal (m²), os lados a e b (m), o diâmetro equivalente, a perda de carga
    por metro e o número de Reynolds.
    """
    if velocity <= 0:
        raise ValueError("A velocidade deve ser positiva.")
    if aspect <= 0:
        raise ValueError("O rácio de forma deve ser positivo.")
    q_m = flow / 3600.0
    area = q_m / velocity
    # Cálculo dos lados a (largura) e b (altura) dados A = a·b e a/b = aspect
    b = math.sqrt(area / aspect)
    a = area / b
    # Diâmetro equivalente (aproximação de ASHRAE)
    # Deq = 1.3 · (A^0.625)/(a + b)^0.25 = 1.3 · A^0.375 / (sqrt(r) + 1/sqrt(r))^0.25
    sqrt_r = math.sqrt(aspect)
    denom = (sqrt_r + 1.0 / sqrt_r) ** 0.25
    deq = 1.3 * (area ** 0.375) / denom
    dp_per_m = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity, deq)
    Re = compute_reynolds(RHO_AIR, velocity, deq)
    return {
        'a': a,
        'b': b,
        'area': area,
        'deq': deq,
        'velocity': velocity,
        'dp_per_m': dp_per_m,
        'Re': Re
    }

def compute_rectangular_from_dp(flow: float, dp_per_m_target: float, aspect: float) -> Dict[str, float]:
    """Dimensiona uma conduta retangular a partir do caudal, perda de carga alvo e rácio de forma.

    Resolve a área de forma iterativa (bissecção), assumindo que a perda de carga
    distribuída por metro obedece à equação de Darcy-Weisbach usando o diâmetro
    equivalente. A função devolve as dimensões a, b, a área, o diâmetro equivalente,
    a velocidade e a perda de carga calculada.
    """
    if dp_per_m_target <= 0:
        raise ValueError("A perda de carga deve ser positiva.")
    if aspect <= 0:
        raise ValueError("O rácio de forma deve ser positivo.")
    q_m = flow / 3600.0
    # Função para calcular Δp/m para uma determinada área A
    def dp_for_area(area: float) -> float:
        if area <= 0:
            return float('inf')
        b = math.sqrt(area / aspect)
        a = area / b
        sqrt_r = math.sqrt(aspect)
        denom = (sqrt_r + 1.0 / sqrt_r) ** 0.25
        deq = 1.3 * (area ** 0.375) / denom
        velocity = q_m / area
        return friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity, deq)
    # Procura bissectiva para encontrar a área cujo Δp/m coincide com o alvo
    # Limites iniciais de procura (m²)
    area_low = 1e-6
    area_high = 10.0  # grande o suficiente para caudais usuais
    # Ajuste de limite superior se Δp(a_high) for demasiado grande
    for _ in range(50):
        if dp_for_area(area_high) < dp_per_m_target:
            break
        area_high *= 2.0
    # Bissecção
    for _ in range(60):  # 60 iterações conferem precisão suficiente
        area_mid = 0.5 * (area_low + area_high)
        dp_mid = dp_for_area(area_mid)
        if dp_mid > dp_per_m_target:
            # Área demasiado pequena (velocidade grande, Δp alta)
            area_low = area_mid
        else:
            area_high = area_mid
    area = area_high
    b = math.sqrt(area / aspect)
    a = area / b
    sqrt_r = math.sqrt(aspect)
    denom = (sqrt_r + 1.0 / sqrt_r) ** 0.25
    deq = 1.3 * (area ** 0.375) / denom
    velocity = q_m / area
    dp_per_m_calc = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity, deq)
    Re = compute_reynolds(RHO_AIR, velocity, deq)
    return {
        'a': a,
        'b': b,
        'area': area,
        'deq': deq,
        'velocity': velocity,
        'dp_per_m': dp_per_m_calc,
        'Re': Re
    }

def area_from_shape(shape: str, diameter: float = None, width: float = None, height: float = None) -> float:
    """Calcula a área de secção transversal dependendo do tipo de conduta.

    Para conduta circular, necessita do diâmetro.
    Para conduta retangular, necessita da largura e da altura.
    """
    if shape == 'circular':
        if diameter is None:
            raise ValueError("Para condutas circulares é necessário fornecer o diâmetro.")
        return math.pi * (diameter ** 2) / 4.0
    elif shape == 'rectangular':
        if width is None or height is None:
            raise ValueError("Para condutas retangulares é necessário fornecer largura e altura.")
        return width * height
    else:
        raise ValueError("Tipo de conduta desconhecido.")


def simulate_network(flow: float, shape: str, initial_diameter: float = None,
                     width: float = None, height: float = None,
                     segments: List[Dict[str, Any]] = None,
                     output_html: str = 'relatorio.html',
                     output_svg: str = 'rede.svg') -> Dict[str, Any]:
    """Simula uma rede de condutas sem ramificações.

    O caudal permanece constante ao longo de toda a rede. Para cada segmento,
    calcula-se a perda de carga distribuída e local (caso aplicável). São
    suportados três tipos de segmentos:

    - `straight` com propriedade `L` (comprimento em metros).
    - `elbow90` ou `elbow45` (curvas a 90° ou 45°), com coeficientes de perda
      pré-definidos.
    - `reducer` com propriedade `D2` (novo diâmetro para condutas circulares) ou
      `width2`/`height2` para condutas retangulares.

    O relatório é gravado em HTML com um desenho SVG da rede incorporado.
    Retorna um dicionário com o resumo dos cálculos e a perda de carga total.
    """
    if segments is None:
        segments = []
    if shape not in ('circular', 'rectangular'):
        raise ValueError("Tipo de conduta inválido para a rede.")
    q_m = flow / 3600.0
    results = []
    # Dimensões actuais
    if shape == 'circular':
        if initial_diameter is None:
            raise ValueError("Para rede circular é necessário fornecer o diâmetro inicial.")
        current_diameter = initial_diameter
        current_width = None
        current_height = None
    else:  # rectangular
        if width is None or height is None:
            raise ValueError("Para rede retangular é necessário fornecer largura e altura iniciais.")
        current_diameter = None
        current_width = width
        current_height = height
    total_dp = 0.0
    # Preparação do SVG
    svg_elements: List[str] = []
    # factores de escala para o desenho (ajustáveis)
    length_scale = 50.0  # pixels por metro de comprimento
    width_scale = 200.0  # pixels por metro de diâmetro
    # Posição inicial do desenho
    x, y = 10.0, 50.0
    # Altura máxima utilizada para ajustar o tamanho da imagem
    max_x, max_y = x, y
    for idx, seg in enumerate(segments):
        seg_type = seg.get('type')
        if seg_type == 'straight':
            L = float(seg.get('L', 0.0))
            if L <= 0:
                raise ValueError("Comprimento do segmento recto deve ser positivo.")
            area = area_from_shape(shape, current_diameter, current_width, current_height)
            velocity = q_m / area
            dp = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, velocity,
                                        current_diameter if shape == 'circular' else compute_equivalent_diameter(current_width, current_height)) * L
            total_dp += dp
            results.append({'segment': idx + 1, 'type': 'straight', 'length': L,
                            'diameter_or_width': current_diameter if shape == 'circular' else current_width,
                            'height': None if shape == 'circular' else current_height,
                            'velocity': velocity, 'dp': dp})
            # Desenho: rectângulo horizontal com comprimento proporcional e largura proporcional ao diâmetro/altura
            if shape == 'circular':
                w_px = current_diameter * width_scale
            else:
                w_px = current_height * width_scale
            h_px = w_px  # a espessura desenhada é representativa da altura/diâmetro
            l_px = L * length_scale
            # Rectângulo (representado como rect)
            svg_elements.append(f'<rect x="{x:.1f}" y="{y - h_px/2:.1f}" width="{l_px:.1f}" height="{h_px:.1f}" '
                                f'style="fill:none;stroke:black;stroke-width:1"/>')
            x += l_px  # avança horizontalmente
            max_x = max(max_x, x)
            max_y = max(max_y, y + h_px / 2)
        elif seg_type == 'elbow90' or seg_type == 'elbow45':
            # Coeficientes de perda típicos: valores aproximados
            zeta = 1.5 if seg_type == 'elbow90' else 0.3
            area = area_from_shape(shape, current_diameter, current_width, current_height)
            velocity = q_m / area
            dp = zeta * (RHO_AIR * velocity * velocity) / 2.0
            total_dp += dp
            results.append({'segment': idx + 1, 'type': seg_type,
                            'diameter_or_width': current_diameter if shape == 'circular' else current_width,
                            'height': None if shape == 'circular' else current_height,
                            'velocity': velocity, 'dp': dp})
            # Desenho: arco de 90° ou 45°
            if shape == 'circular':
                r_px = current_diameter * width_scale  # raio proporcional
            else:
                r_px = current_height * width_scale
            sweep = 90.0 if seg_type == 'elbow90' else 45.0
            # Desenhar arco com SVG path: 'M'ove + 'A'rc. A âncora do arco parte do ponto actual.
            # Ajuste: desenha um quarto de círculo com raio r_px. O início do arco é no fim da última recta.
            # Neste modelo simples, consideramos um arco na horizontal que vira para cima (90°) ou para cima/diagonal (45°).
            large_arc_flag = 0
            sweep_flag = 1  # sentido horário
            # Fim do arco
            if seg_type == 'elbow90':
                end_x = x
                end_y = y - r_px
            else:  # 45°
                end_x = x + r_px / math.sqrt(2)
                end_y = y - r_px / math.sqrt(2)
            svg_elements.append(
                f'<path d="M {x:.1f},{y:.1f} A {r_px:.1f},{r_px:.1f} 0 {large_arc_flag} {sweep_flag} {end_x:.1f},{end_y:.1f}" '
                f'style="fill:none;stroke:black;stroke-width:1"/>'
            )
            # Actualiza a posição final para o próximo segmento
            x, y = end_x, end_y
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        elif seg_type == 'reducer':
            # Redutor: altera a dimensão transversal. Para condutas circulares utiliza `D2`,
            # para retangulares utiliza `width2` e `height2`.
            if shape == 'circular':
                new_d = seg.get('D2')
                if new_d is None or new_d <= 0:
                    raise ValueError("Segmento reducer requer propriedade D2 > 0 para condutas circulares.")
                area1 = area_from_shape('circular', current_diameter)
                area2 = area_from_shape('circular', new_d)
                v1 = q_m / area1
                v2 = q_m / area2
                # coeficiente local aproximado para redutor abrupto
                zeta = 0.5
                dp = zeta * (RHO_AIR * v2 * v2) / 2.0
                total_dp += dp
                results.append({'segment': idx + 1, 'type': 'reducer', 'diameter_or_width': current_diameter,
                                'height': None, 'velocity': v1, 'dp': dp, 'new_diameter': new_d})
                # Desenho: transição linear de largura
                w1_px = current_diameter * width_scale
                w2_px = new_d * width_scale
                l_px = 1.0 * length_scale  # comprimento simbólico de 1 m
                svg_elements.append(
                    f'<polygon points="{x:.1f},{y - w1_px/2:.1f} {x + l_px:.1f},{y - w2_px/2:.1f} '
                    f'{x + l_px:.1f},{y + w2_px/2:.1f} {x:.1f},{y + w1_px/2:.1f}" '
                    f'style="fill:none;stroke:black;stroke-width:1"/>'
                )
                x += l_px
                current_diameter = new_d
                max_x = max(max_x, x)
                max_y = max(max_y, y + max(w1_px/2, w2_px/2))
            else:
                new_w = seg.get('width2')
                new_h = seg.get('height2')
                if new_w is None or new_h is None or new_w <= 0 or new_h <= 0:
                    raise ValueError("Segmento reducer requer width2 e height2 > 0 para condutas retangulares.")
                area1 = area_from_shape('rectangular', None, current_width, current_height)
                area2 = area_from_shape('rectangular', None, new_w, new_h)
                v1 = q_m / area1
                v2 = q_m / area2
                zeta = 0.5
                dp = zeta * (RHO_AIR * v2 * v2) / 2.0
                total_dp += dp
                results.append({'segment': idx + 1, 'type': 'reducer', 'diameter_or_width': current_width,
                                'height': current_height, 'velocity': v1, 'dp': dp,
                                'new_width': new_w, 'new_height': new_h})
                # Desenho: transição de rectângulo
                w1_px = current_height * width_scale
                w2_px = new_h * width_scale
                l_px = 1.0 * length_scale
                svg_elements.append(
                    f'<polygon points="{x:.1f},{y - w1_px/2:.1f} {x + l_px:.1f},{y - w2_px/2:.1f} '
                    f'{x + l_px:.1f},{y + w2_px/2:.1f} {x:.1f},{y + w1_px/2:.1f}" '
                    f'style="fill:none;stroke:black;stroke-width:1"/>'
                )
                x += l_px
                current_width = new_w
                current_height = new_h
                max_x = max(max_x, x)
                max_y = max(max_y, y + max(w1_px/2, w2_px/2))
        else:
            raise ValueError(f"Tipo de segmento desconhecido: {seg_type}")
    # Encapsula os elementos de desenho no SVG final
    width_svg = max_x + 10.0
    height_svg = max_y + 50.0
    # Cabeçalho do SVG incluindo rectângulo de fundo branco. A construção usa
    # parênteses para evitar a necessidade de quebras de linha com "\".
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_svg:.1f}" height="{height_svg:.1f}">'  # abertura
        f'<rect x="0" y="0" width="{width_svg:.1f}" height="{height_svg:.1f}" fill="white" stroke="none"/>'
    )
    svg_body = "\n".join(svg_elements)
    svg_footer = '</svg>'
    svg_content = svg_header + "\n" + svg_body + "\n" + svg_footer
    # Gravar ficheiro SVG
    with open(output_svg, 'w', encoding='utf-8') as f_svg:
        f_svg.write(svg_content)
    # Construir HTML
    html_parts: List[str] = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Relatório de Rede de Condutas</title></head><body>")
    html_parts.append("<h1>Relatório de Rede de Condutas</h1>")
    html_parts.append(f"<p>Caudal: {flow:.2f} m³/h</p>")
    html_parts.append(f"<p>Perda de carga total: {total_dp:.3f} Pa</p>")
    # Tabela de resultados por segmento
    html_parts.append("<table border='1' cellspacing='0' cellpadding='4'>")
    # Cabeçalho
    if shape == 'circular':
        html_parts.append("<tr><th>#</th><th>Tipo</th><th>Diâmetro (m)</th><th>Velocidade (m/s)</th><th>Δp (Pa)</th></tr>")
        for r in results:
            d = r.get('diameter_or_width')
            v = r.get('velocity')
            dp = r.get('dp')
            seg_type = r.get('type')
            html_parts.append(f"<tr><td>{r['segment']}</td><td>{seg_type}</td><td>{d:.3f}</td><td>{v:.3f}</td><td>{dp:.3f}</td></tr>")
    else:
        html_parts.append("<tr><th>#</th><th>Tipo</th><th>Largura (m)</th><th>Altura (m)</th><th>Velocidade (m/s)</th><th>Δp (Pa)</th></tr>")
        for r in results:
            w = r.get('diameter_or_width')
            h = r.get('height')
            v = r.get('velocity')
            dp = r.get('dp')
            seg_type = r.get('type')
            html_parts.append(f"<tr><td>{r['segment']}</td><td>{seg_type}</td><td>{w:.3f}</td><td>{h:.3f}</td><td>{v:.3f}</td><td>{dp:.3f}</td></tr>")
    html_parts.append("</table>")
    # Incorporar o SVG
    html_parts.append("<h2>Esquema da Rede</h2>")
    # Incorporar diretamente o conteúdo SVG
    encoded_svg = svg_content.replace("\n", "")
    html_parts.append(f"<div>{encoded_svg}</div>")
    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)
    with open(output_html, 'w', encoding='utf-8') as f_html:
        f_html.write(html_content)
    return {'total_dp': total_dp, 'segments': results}

def compute_equivalent_diameter(width: float, height: float) -> float:
    """Calcula o diâmetro equivalente de uma conduta retangular.

    Fórmula de ASHRAE/EN (aproximação):
        Deq = 1.3 · (A^0.625)/(a + b)^0.25
        = 1.3 · A^0.375 / (sqrt(r) + 1/sqrt(r))^0.25
    onde r = a/b.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Dimensões devem ser positivas.")
    area = width * height
    r = width / height
    sqrt_r = math.sqrt(r)
    denom = (sqrt_r + 1.0 / sqrt_r) ** 0.25
    return 1.3 * (area ** 0.375) / denom

def run_tests() -> None:
    """Executa uma bateria de testes unitários rudimentares para validar as funções.

    Lança exceção em caso de falha. Caso todos os testes passem, imprime uma
    mensagem indicando sucesso.
    """
    # Teste 1: conduta circular a partir da velocidade
    res1 = compute_circular_from_velocity(1800.0, 5.0)
    expected_d1 = math.sqrt(4.0 * ((1800/3600.0)/5.0) / math.pi)
    assert abs(res1['diameter'] - expected_d1) < 1e-6, f"Teste 1 falhou: diâmetro {res1['diameter']:.6f} != {expected_d1:.6f}"
    # Verificar perda de carga aproximada
    dp_expected = friction_pressure_drop(FRICTION_FACTOR, RHO_AIR, 5.0, res1['diameter'])
    assert abs(res1['dp_per_m'] - dp_expected) < 1e-9
    # Teste 2: conduta circular a partir de Δp
    res2 = compute_circular_from_dp(1800.0, 0.9)
    # Compara que a perda de carga obtida está próxima da meta
    assert abs(res2['dp_per_m'] - 0.9) < 0.05, f"Teste 2 falhou: Δp {res2['dp_per_m']:.3f} Pa/m != 0.9 Pa/m"
    # Teste 3: conduta retangular a partir da velocidade (rácio 2:1)
    res3 = compute_rectangular_from_velocity(2500.0, 5.0, 2.0)
    q_m3 = 2500.0/3600.0
    area3 = q_m3/5.0
    b3 = math.sqrt(area3/2.0)
    a3 = area3/b3
    assert abs(res3['a'] - a3) < 1e-6, "Teste 3 falhou na dimensão a"
    assert abs(res3['b'] - b3) < 1e-6, "Teste 3 falhou na dimensão b"
    # Teste 4: conduta retangular a partir da Δp
    res4 = compute_rectangular_from_dp(2500.0, res3['dp_per_m'], 2.0)
    # O resultado deve aproximar-se do mesmo diâmetro equivalente e mesma área
    assert abs(res4['deq'] - res3['deq']) < 1e-3, "Teste 4 falhou no diâmetro equivalente"
    # Teste 5: simulação de rede simples circular
    segs = [
        {'type': 'straight', 'L': 10},
        {'type': 'elbow90'},
        {'type': 'straight', 'L': 5}
    ]
    sim = simulate_network(2000.0, 'circular', initial_diameter=0.3, segments=segs, output_html='test.html', output_svg='test.svg')
    assert 'total_dp' in sim and sim['total_dp'] > 0, "Teste 5 falhou: perda de carga total inválida"
    # Certifique-se que os ficheiros foram criados
    assert os.path.exists('test.html'), "Teste 5 falhou: ficheiro HTML não criado"
    assert os.path.exists('test.svg'), "Teste 5 falhou: ficheiro SVG não criado"
    # Limpar ficheiros de teste
    os.remove('test.html')
    os.remove('test.svg')
    print("Todos os testes passaram com sucesso.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculadora e simulador de condutas AVAC")
    parser.add_argument('--test', action='store_true', help='Executa a bateria de testes unitários')
    subparsers = parser.add_subparsers(dest='command')
    # Subcomando select
    parser_select = subparsers.add_parser('select', help='Dimensionamento de condutas (circular ou retangular)')
    parser_select.add_argument('--shape', choices=['circular', 'rectangular'], required=True, help='Tipo de conduta')
    parser_select.add_argument('--flow', type=float, required=True, help='Caudal (m³/h)')
    parser_select.add_argument('--vel', type=float, help='Velocidade (m/s)')
    parser_select.add_argument('--dp_per_m', type=float, help='Perda de carga por metro (Pa/m)')
    parser_select.add_argument('--aspect', type=float, default=1.0, help='Rácio a/b para condutas retangulares (a>=b)')
    # Subcomando network
    parser_net = subparsers.add_parser('network', help='Simulação de rede de condutas 2D')
    parser_net.add_argument('--shape', choices=['circular', 'rectangular'], required=True, help='Tipo de conduta')
    parser_net.add_argument('--flow', type=float, required=True, help='Caudal (m³/h)')
    parser_net.add_argument('--diameter', type=float, help='Diâmetro inicial (m) para condutas circulares')
    parser_net.add_argument('--width', type=float, help='Largura inicial (m) para condutas retangulares')
    parser_net.add_argument('--height', type=float, help='Altura inicial (m) para condutas retangulares')
    parser_net.add_argument('--segments', type=str, required=True, help='JSON com lista de segmentos da rede')
    args = parser.parse_args()
    if args.test:
        run_tests()
        return
    if args.command == 'select':
        if args.vel is None and args.dp_per_m is None:
            parser.error("É necessário fornecer --vel ou --dp_per_m para o dimensionamento.")
        if args.vel is not None and args.dp_per_m is not None:
            parser.error("Deve escolher apenas um critério: --vel ou --dp_per_m.")
        if args.shape == 'circular':
            if args.vel is not None:
                res = compute_circular_from_velocity(args.flow, args.vel)
            else:
                res = compute_circular_from_dp(args.flow, args.dp_per_m)
            # Apresentação dos resultados
            print("Dimensionamento de conduta circular:")
            print(f"Caudal: {args.flow:.2f} m³/h")
            print(f"Diâmetro: {res['diameter']:.3f} m")
            print(f"Área: {res['area']:.4f} m²")
            print(f"Velocidade: {res['velocity']:.3f} m/s")
            print(f"Perda de carga por metro: {res['dp_per_m']:.3f} Pa/m")
            print(f"Número de Reynolds: {res['Re']:.1f}")
        else:  # retangular
            if args.vel is not None:
                res = compute_rectangular_from_velocity(args.flow, args.vel, args.aspect)
            else:
                res = compute_rectangular_from_dp(args.flow, args.dp_per_m, args.aspect)
            print("Dimensionamento de conduta retangular:")
            print(f"Caudal: {args.flow:.2f} m³/h")
            print(f"Largura a: {res['a']:.3f} m")
            print(f"Altura b: {res['b']:.3f} m")
            print(f"Área: {res['area']:.4f} m²")
            print(f"Diâmetro equivalente: {res['deq']:.3f} m")
            print(f"Velocidade: {res['velocity']:.3f} m/s")
            print(f"Perda de carga por metro: {res['dp_per_m']:.3f} Pa/m")
            print(f"Número de Reynolds: {res['Re']:.1f}")
    elif args.command == 'network':
        # Conversão da string JSON de segmentos em lista de dicionários
        try:
            segments = json.loads(args.segments)
            if not isinstance(segments, list):
                raise ValueError
        except Exception:
            print("Erro: --segments deve ser uma lista JSON de segmentos.")
            sys.exit(1)
        if args.shape == 'circular':
            if args.diameter is None:
                print("Erro: --diameter é obrigatório para rede circular.")
                sys.exit(1)
            res = simulate_network(args.flow, 'circular', initial_diameter=args.diameter, segments=segments)
        else:
            if args.width is None or args.height is None:
                print("Erro: --width e --height são obrigatórios para rede retangular.")
                sys.exit(1)
            res = simulate_network(args.flow, 'rectangular', width=args.width, height=args.height, segments=segments)
        print(f"Perda de carga total na rede: {res['total_dp']:.3f} Pa")
        print("Relatório gerado: relatorio.html (com desenho SVG)")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()