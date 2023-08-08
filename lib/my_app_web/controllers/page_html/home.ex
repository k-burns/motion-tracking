defmodule MyAppWeb.HomeLive.Index do
  use MyAppWeb, :live_view

  @impl true
  def mount(_params, _, socket) do
    {:ok,
     socket
     |> assign(running?: false)
     |> assign(image: nil)
     |> assign(prediction: nil)
     |> assign(serving: serving())
     |> assign(
       video:
         Evision.VideoCapture.videoCapture(
           "/Users/katelynnburns/Documents/Zoom/2023-07-25 10.51.03 Katelynn (she_her)'s Zoom Meeting/video1536544321.mp4"
         )
     )}
  end

  @impl true

  def handle_event("start", _params, socket) do
    send(self(), :run)

    {:noreply, assign(socket, running?: true)}
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
       |> assign(prediction: prediction)
      }
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
    tensor = frame |> Evision.cvtColor(7) |> Evision.gaussianBlur({23, 23}, 30)

    {contours, _} =
      Evision.findContours(
        tensor,
        Evision.Constant.cv_RETR_LIST(),
        Evision.Constant.cv_CHAIN_APPROX_NONE()
      )

    # color in {Blue, Green, Red}, range from 0-255
    edge_color = {0, 0, 255}

    # # draw all contours by setting `index` to `-1`
    index = -1

    # # Load image in color

    # # draw all contours on the color image
    Evision.drawContours(frame, contours, index, edge_color, thickness: 2)
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
