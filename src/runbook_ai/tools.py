        raise ToolInputError("Use a non-empty arithmetic expression of at most 120 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ToolInputError("That is not a valid arithmetic expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > 32:
        raise ToolInputError("The expression is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1_000_000_000:
                raise ToolInputError("Numbers must be smaller than one billion in magnitude.")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 10_000):
                raise ToolInputError("Powers are limited to an exponent of 8 and a base of 10,000.")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if isinstance(result, complex) or not math.isfinite(result) or abs(result) > 1e12:
                raise ToolInputError("The result is outside the supported range.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ToolInputError("Only basic arithmetic is supported.")

    try:
        return evaluate(tree)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ToolInputError("That expression cannot be evaluated safely.") from exc


def _search_docs(args: dict, retriever) -> dict:
    _require_keys(args, {"query"}, {"top_k"})
    query = args["query"]
    top_k = args.get("top_k", 5)
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ToolInputError("Search query must contain 1 to 1,000 characters.")
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ToolInputError("top_k must be an integer from 1 to 5.")
    matches = []
    for chunk, score in retriever.search(query.strip(), k=top_k):
        if score < MIN_RELEVANCE:
            continue
        matches.append({
            "source": chunk.source,
            "section": chunk.section,
            "page": chunk.page,
            "excerpt": chunk.text[:1200],
            "similarity": round(float(score), 4),
        })
    return {"matches": matches, "query": query.strip()}


def execute_tool(name: str, arguments, retriever=None) -> dict:
    """Run one explicitly allowlisted tool after validating all arguments."""
    if name not in ALLOWED_TOOLS:
        return {"error": "Tool is not allowlisted."}
    args = _require_object(arguments)
    if name == "search_docs":
        if retriever is None:
            return {"matches": [], "query": args.get("query", "")}
        return _search_docs(args, retriever)
    if name == "calculator":
        _require_keys(args, {"expression"})
        return {"expression": args["expression"], "result": safe_calculate(args["expression"])}
    if name == "get_date_info":
        _require_keys(args, set(), {"date"})
        requested = args.get("date", "today")
        if not isinstance(requested, str) or len(requested) > 32:
            raise ToolInputError("Use 'today' or an ISO date such as 2026-10-26.")
        if requested.strip().lower() == "today":
            local_now = datetime.now().astimezone()
            return {
                "date": local_now.date().isoformat(),
                "weekday": local_now.strftime("%A"),
                "timezone": local_now.tzname() or "local time",
            }
        try:
            parsed = date.fromisoformat(requested.strip())
        except ValueError as exc:
            raise ToolInputError("Use 'today' or an ISO date such as 2026-10-26.") from exc
        return {"date": parsed.isoformat(), "weekday": parsed.strftime("%A")}
    if name == "create_ticket_draft":
        _require_keys(args, {"title", "description", "category"})
        title, description, category = args["title"], args["description"], args["category"]
        if not isinstance(title, str) or not 3 <= len(title.strip()) <= 120:
            raise ToolInputError("Ticket title must contain 3 to 120 characters.")
        if not isinstance(description, str) or not 3 <= len(description.strip()) <= 1200:
            raise ToolInputError("Ticket details must contain 3 to 1,200 characters.")
        if not isinstance(category, str) or category not in {
            "account", "network", "hardware", "software", "other"
        }:
            raise ToolInputError("Choose a supported ticket category.")
        return {
            "ticket_id": f"RB-{secrets.token_hex(3).upper()}",
            "title": title.strip(),
            "description": description.strip(),
            "category": category,
            "status": "Draft - not submitted",
        }
    if name == "get_local_status":
        _require_keys(args, set())
        status = inspect_ollama()
        return {
            "ollama": "online" if status.online else "offline",
            "chat_model": CHAT_MODEL,
            "chat_model_available": status.chat_ready,
            "embedding_model": EMBED_MODEL,
            "embedding_model_available": status.embedding_ready,
        }
    return {"error": "Tool is not allowlisted."}
