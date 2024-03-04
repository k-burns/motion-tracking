defmodule MyApp.Events do
  def listen_for_event(module, event_name) do
    Registry.register(
      __MODULE__,
      event_name,
      {module, event_name}
    )
  end

  def notify(event_name, arguments) do
    Registry.dispatch(__MODULE__, event_name, fn entries ->
      for {pid, _} <- entries, do: send(pid, {event_name, arguments})
    end)
  end
end
