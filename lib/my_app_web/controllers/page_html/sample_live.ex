defmodule MyApp.SampleLive do
  use Phoenix.LiveView, layout: {PhoenixDemo.Layouts, :live}

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(label: nil, task: nil)
     |> allow_upload(:image, accept: :any, progress: &handle_progress/3, auto_upload: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="h-screen w-screen flex items-center justify-center antialiased">
      <div class="flex flex-col items-center w-1/2">
        <form class="m-0 flex flex-col items-center space-y-2" phx-change="noop" phx-submit="noop">
          <.image_input id="image" upload={@uploads.image} height={224} width={224} />
        </form>
        <div class="mt-6 flex space-x-1.5 items-center text-gray-600 text-lg">
          <span>Label:</span>
          <%= if @task do %>
            <.spinner />
          <% else %>
            <span class="text-gray-900 font-medium"><%= @label || "?" %></span>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp image_input(assigns) do
    ~H"""
    <div
      id={"#{@id}-root"}
      class="inline-flex p-4 border-2 border-dashed border-gray-200 rounded-lg cursor-pointer"
      phx-hook="ImageInput"
      data-height={@height}
      data-width={@width}
    >
      <.live_file_input upload={@upload} class="hidden" />
      <input type="file" data-el-input class="hidden" />
      <div
        class="h-[300px] w-[300px] flex items-center justify-center"
        id={"#{@id}-preview"}
        phx-update="ignore"
        data-el-preview
      >
        <div class="text-gray-500 text-center">
          Drag an image file here or click to open file browser
        </div>
      </div>
    </div>
    """
  end

  defp spinner(assigns) do
    ~H"""
    <svg
      class="inline mr-2 w-4 h-4 text-gray-200 animate-spin fill-blue-600"
      viewBox="0 0 100 101"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
        fill="currentColor"
      />
      <path
        d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
        fill="currentFill"
      />
    </svg>
    """
  end

  def handle_progress(:image, entry, socket) when entry.done? do
    binary =
      consume_uploaded_entry(socket, entry, fn %{path: path} ->
        {:ok, File.read!(path)}
      end)

    image = decode_as_tensor(binary)
    task = Task.async(fn -> Nx.Serving.batched_run(PhoenixDemo.Serving, image) end)
    {:noreply, assign(socket, task: task)}
  end

  def handle_progress(_name, _entry, socket), do: {:noreply, socket}

  defp decode_as_tensor(<<height::32-integer, width::32-integer, data::binary>>) do
    data |> Nx.from_binary(:u8) |> Nx.reshape({height, width, 3})
  end

  @impl true
  def handle_event("noop", %{}, socket) do
    # We need phx-change and phx-submit on the form for live uploads,
    # but we make predictions immediately using :progress, so we just
    # ignore this event
    {:noreply, socket}
  end

  @impl true
  def handle_info({ref, result}, socket) when socket.assigns.task.ref == ref do
    Process.demonitor(ref, [:flush])
    %{predictions: [%{label: label}]} = result
    {:noreply, assign(socket, label: label, task: nil)}
  end
end
