defmodule MyAppWeb.HomeLive.Index do
  use MyAppWeb, :live_view
  alias Phoenix.HTML.Form

  @impl true
  def mount(_params, _, socket) do
    {:ok,
     socket
     |> assign(running?: false)
     |> assign(image: nil)
     |> assign(prediction: nil)
     |> assign(serving: serving())
     |> assign(video: nil)}
  end

  @impl true

  def handle_event("start", %{"video_input" => %{"video_path" => path}}, socket) do
    send(self(), :run)

    {:noreply, assign(socket, running?: true, video: Evision.VideoCapture.videoCapture(path))}
  end

  @impl true

  def handle_event("start_camera", _params, socket) do
    send(self(), :run)

    {:noreply, assign(socket, running?: true, video: Evision.VideoCapture.videoCapture())}
  end

  @impl true

  def handle_info(:run, %{assigns: %{running?: true, prediction: nil}} = socket) do
    frame = socket.assigns.video |> Evision.VideoCapture.read()

    myimage_grey =
      Evision.cvtColor(frame, Evision.Constant.cv_COLOR_BGR2GRAY())
      |> Evision.gaussianBlur({23, 23}, 30)

    {_ret, baseline} =
      Evision.threshold(myimage_grey, 127, 255, Evision.Constant.cv_THRESH_TRUNC())

    {_ret, background} =
      Evision.threshold(baseline, 126, 255, Evision.Constant.cv_THRESH_BINARY())

    {contours, _} =
      Evision.findContours(
        background,
        Evision.Constant.cv_RETR_LIST(),
        Evision.Constant.cv_CHAIN_APPROX_NONE()
      )

    if length(contours) > 0 do
      [contour | _contours] =
        Enum.sort(contours, &(&1 |> Evision.contourArea() >= &2 |> Evision.contourArea()))

      clone = Evision.Mat.clone(frame)
      foreground = clone |> Evision.fillPoly([contour], {255, 255, 255})
      prediction = predict(socket.assigns.serving, foreground)

      send(self(), :run)

      {:noreply,
       socket
       |> assign(prediction: prediction)}
    else
      send(self(), :run)
      {:noreply, socket}
    end
  end

  @impl true

  def handle_info(:run, %{assigns: %{running?: true}} = socket) do
    frame = socket.assigns.video |> Evision.VideoCapture.read()

    image = track(frame)

    send(self(), :run)

    {:noreply,
     socket
     |> assign(image: Evision.imencode(".jpg", image) |> Base.encode64())}
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  defp predict(serving, frame) do
    pred_tensor = frame |> Evision.Mat.to_nx() |> Nx.backend_transfer()
    %{predictions: [%{label: label}]} = Nx.Serving.run(serving, pred_tensor)

    label
  end

  defp track(frame) do
    tensor =
      frame
      |> Evision.gaussianBlur({7, 7}, 1)
      |> Evision.cvtColor(Evision.Constant.cv_COLOR_BGR2GRAY())

    background = Evision.adaptiveThreshold(tensor, 255, 0, 0, 25, 10)
    kernel = Evision.Mat.ones({5, 5}, :u8)
    background = Evision.erode(background, kernel, iterations: 1)
    background = Evision.morphologyEx(background, Evision.Constant.cv_MORPH_OPEN(), kernel)

    {contours, _} =
      Evision.findContours(
        background,
        Evision.Constant.cv_RETR_TREE(),
        Evision.Constant.cv_CHAIN_APPROX_SIMPLE()
      )

    minimal_area = 50000

    contours =
      Enum.reject(contours, fn c ->
        area = Evision.contourArea(c)

        area < minimal_area
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

  defp serving do
    {:ok, model_info} = Bumblebee.load_model({:hf, "facebook/convnext-tiny-224"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "facebook/convnext-tiny-224"})

    Bumblebee.Vision.image_classification(model_info, featurizer,
      top_k: 1,
      compile: [batch_size: 1],
      defn_options: [compiler: EXLA]
    )
  end
end
