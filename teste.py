from graphviz import Digraph


def add_task(dot, node_id, label, fillcolor):
    dot.node(
        node_id,
        label,
        shape="box",
        style="rounded,filled",
        fillcolor=fillcolor,
        color="black",
        fontname="Arial",
        fontsize="10",
        margin="0.12,0.08",
    )


def add_gateway(dot, node_id, label):
    dot.node(
        node_id,
        label,
        shape="diamond",
        style="filled",
        fillcolor="#FFF2CC",
        color="black",
        fontname="Arial",
        fontsize="10",
        margin="0.08,0.05",
    )


def add_event(dot, node_id, label, event_type="start"):
    shape = "circle" if event_type == "start" else "doublecircle"
    color = "#D5E8D4" if event_type == "start" else "#F8CECC"

    dot.node(
        node_id,
        label,
        shape=shape,
        style="filled",
        fillcolor=color,
        color="black",
        fontname="Arial",
        fontsize="10",
        width="0.9",
        height="0.9",
    )


def edge(dot, a, b, label=None, style="solid", constraint="true"):
    kwargs = {
        "style": style,
        "fontname": "Arial",
        "fontsize": "10",
        "constraint": constraint,
    }
    if label is not None:
        kwargs["label"] = label
    dot.edge(a, b, **kwargs)


def build_process_model():
    dot = Digraph("processo_manutencao_computadores", format="png")

    # Layout geral: mais vertical e compacto
    dot.attr(
        rankdir="TB",          # top -> bottom
        splines="ortho",
        newrank="true",
        nodesep="0.45",
        ranksep="0.55",
        pad="0.2",
        fontname="Arial",
        labelloc="t",
        label="Processo de Manutenção de Computadores",
    )

    # Cores por setor
    funcionario_color = "#DAE8FC"
    ti_color = "#E1D5E7"
    almox_color = "#D5E8D4"
    compras_color = "#FCE5CD"
    sistema_color = "#EEEEEE"

    # -------------------------
    # Nós
    # -------------------------

    # Funcionário
    add_event(dot, "start", "Início", "start")
    add_task(dot, "identificar_problema", "Identificar problema\nno computador", funcionario_color)
    add_task(dot, "abrir_chamado", "Abrir chamado no\nsistema de suporte", funcionario_color)
    add_task(dot, "confirmar_funcionamento", "Confirmar funcionamento\nadequado", funcionario_color)

    # TI
    add_task(dot, "receber_chamado", "Receber chamado", ti_color)
    add_task(dot, "analise_inicial", "Realizar análise\ninicial do problema", ti_color)
    add_gateway(dot, "g_remoto", "Pode ser resolvido\nremotamente?")
    add_task(dot, "atendimento_online", "Realizar atendimento\nonline", ti_color)
    add_task(dot, "verificar_funcionamento", "Verificar se o equipamento\nvoltou a funcionar", ti_color)
    add_gateway(dot, "g_resolvido", "Problema\nresolvido?")
    add_task(dot, "agendar_presencial", "Agendar manutenção\npresencial", ti_color)
    add_task(dot, "avaliar_equipamento", "Avaliar equipamento\npresencialmente", ti_color)
    add_gateway(dot, "g_troca_pecas", "Precisa trocar\npeças?")
    add_task(dot, "continuar_manutencao", "Continuar manutenção", ti_color)
    add_task(dot, "realizar_reparo", "Realizar reparo", ti_color)
    add_task(dot, "testar_equipamento", "Testar equipamento", ti_color)

    # Almoxarifado
    add_task(dot, "solicitar_peca", "Solicitar peça ao\nalmoxarifado", almox_color)
    add_task(dot, "verificar_peca", "Verificar disponibilidade\nda peça", almox_color)
    add_gateway(dot, "g_peca_disponivel", "Peça\n disponível?")

    # Compras
    add_task(dot, "comprar_peca", "Realizar compra\nda peça", compras_color)
    add_task(dot, "aguardar_entrega", "Aguardar entrega\nda peça", compras_color)

    # Sistema / registro
    add_task(dot, "finalizar_chamado", "Finalizar chamado", sistema_color)
    add_task(dot, "arquivar_relatorio", "Arquivar relatório\nde manutenção", sistema_color)
    add_event(dot, "end", "Fim", "end")

    # -------------------------
    # Agrupamentos para deixar mais "quadrado"
    # -------------------------

    # Linha 1
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("start")
        s.node("identificar_problema")
        s.node("abrir_chamado")

    # Linha 2
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("receber_chamado")
        s.node("analise_inicial")
        s.node("g_remoto")

    # Linha 3
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("atendimento_online")
        s.node("agendar_presencial")

    # Linha 4
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("verificar_funcionamento")
        s.node("avaliar_equipamento")

    # Linha 5
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("g_resolvido")
        s.node("g_troca_pecas")

    # Linha 6
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("solicitar_peca")
        s.node("realizar_reparo")

    # Linha 7
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("verificar_peca")
        s.node("testar_equipamento")

    # Linha 8
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("g_peca_disponivel")
        s.node("confirmar_funcionamento")

    # Linha 9
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("comprar_peca")
        s.node("continuar_manutencao")
        s.node("finalizar_chamado")

    # Linha 10
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("aguardar_entrega")
        s.node("arquivar_relatorio")
        s.node("end")

    # -------------------------
    # Fluxo principal
    # -------------------------

    edge(dot, "start", "identificar_problema")
    edge(dot, "identificar_problema", "abrir_chamado")
    edge(dot, "abrir_chamado", "receber_chamado")
    edge(dot, "receber_chamado", "analise_inicial")
    edge(dot, "analise_inicial", "g_remoto")

    # Caminho remoto
    edge(dot, "g_remoto", "atendimento_online", "Sim")
    edge(dot, "atendimento_online", "verificar_funcionamento")
    edge(dot, "verificar_funcionamento", "g_resolvido")

    # Remoto resolveu
    edge(dot, "g_resolvido", "finalizar_chamado", "Sim")

    # Remoto não resolveu
    edge(dot, "g_resolvido", "agendar_presencial", "Não")

    # Não pode remoto
    edge(dot, "g_remoto", "agendar_presencial", "Não")

    # Caminho presencial
    edge(dot, "agendar_presencial", "avaliar_equipamento")
    edge(dot, "avaliar_equipamento", "g_troca_pecas")

    # Não precisa peça
    edge(dot, "g_troca_pecas", "realizar_reparo", "Não")

    # Precisa peça
    edge(dot, "g_troca_pecas", "solicitar_peca", "Sim")
    edge(dot, "solicitar_peca", "verificar_peca")
    edge(dot, "verificar_peca", "g_peca_disponivel")

    # Peça disponível
    edge(dot, "g_peca_disponivel", "continuar_manutencao", "Sim")

    # Peça indisponível
    edge(dot, "g_peca_disponivel", "comprar_peca", "Não")
    edge(dot, "comprar_peca", "aguardar_entrega")
    edge(dot, "aguardar_entrega", "continuar_manutencao")

    # Continuação final
    edge(dot, "continuar_manutencao", "realizar_reparo")
    edge(dot, "realizar_reparo", "testar_equipamento")
    edge(dot, "testar_equipamento", "confirmar_funcionamento")
    edge(dot, "confirmar_funcionamento", "finalizar_chamado")
    edge(dot, "finalizar_chamado", "arquivar_relatorio")
    edge(dot, "arquivar_relatorio", "end")

    return dot


if __name__ == "__main__":
    process = build_process_model()

    # Gera PNG
    process.render("processo_manutencao_computadores_organizado", cleanup=True)

    # Gera PDF
    process.format = "pdf"
    process.render("processo_manutencao_computadores_organizado", cleanup=True)

    print("Arquivos gerados:")
    print("- processo_manutencao_computadores_organizado.png")
    print("- processo_manutencao_computadores_organizado.pdf")