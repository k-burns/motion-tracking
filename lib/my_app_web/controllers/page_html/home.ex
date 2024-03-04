defmodule MyAppWeb.HomeLive.Index do
  use MyAppWeb, :live_view
  alias Phoenix.HTML.Form
  alias MyApp.Events

  @impl true
  def mount(_params, _, socket) do
    Phoenix.PubSub.subscribe(MyApp.PubSub, "Prediction")
    {:ok,
     socket
     |> assign(running?: false)
     |> assign(image: nil)
     |> assign(prediction: nil)
     |> assign(video: nil)}
  end

  @impl true

  def handle_event("start", %{"video_input" => %{"video_path" => ""}}, socket) do
    video = Evision.VideoCapture.videoCapture(0)
    send(self(), :run)
    Events.notify(:predict, %{video: video})

    {:noreply, assign(socket, running?: true, video: video)}
  end

  @impl true

  def handle_event("start", %{"video_input" => %{"video_path" => path}}, socket) do
    video = Evision.VideoCapture.videoCapture(path)
    send(self(), :run)
    Events.notify(:predict, %{video: video})


    {:noreply, assign(socket, running?: true, video: video)}
  end

  @impl true

  def handle_info(:run, socket) do
    frame = socket.assigns.video |> Evision.VideoCapture.read()
    image = track(frame)

    send(self(), :run)

   {:noreply,
   socket
   |> assign(image: Evision.imencode(".jpg", image) |> Base.encode64())}

  end

  @impl true

  def handle_info({:prediction, prediction}, socket) do
    Events.notify(:predict, %{video: socket.assigns.video})

   {:noreply,
   socket
   |> assign(prediction: prediction)}

  end

  @impl true

  def handle_info(_msg, socket) do
    {:noreply, socket}
   end

  defp track(frame) do
    contours = find_contours(frame)

    minimal_area = 5000
    maximum_area = 500_000

    contours =
      Enum.reject(contours, fn c ->
        area = Evision.contourArea(c)

        area < minimal_area || area > maximum_area
      end)

    new_frame =
      if contours != [] do
        Enum.reduce(contours, frame, fn c, acc ->
          {x, y, w, h} = Evision.boundingRect(c)
          Evision.rectangle(acc, {x, y}, {x + w, y + h}, {255, 0, 0}, thickness: 4, lineType: 4)
        end)
      else
        frame
      end

    edge_color = {0, 0, 255}

    index = -1

    Evision.drawContours(new_frame, contours, index, edge_color, thickness: 4)
  end

  defp find_contours(frame) do
    myimage_grey =
      Evision.cvtColor(frame, Evision.Constant.cv_COLOR_BGR2GRAY())
      |> Evision.gaussianBlur({23, 23}, 30)

    {_ret, background} =
      Evision.threshold(myimage_grey, 126, 255, Evision.Constant.cv_THRESH_BINARY())

    {contours, _} =
      Evision.findContours(
        background,
        Evision.Constant.cv_RETR_LIST(),
        Evision.Constant.cv_CHAIN_APPROX_NONE()
      )

    contours
  end
end
