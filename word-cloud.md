# Word Cloud

chart = {
  const svg = d3.create("svg")
      .attr("viewBox", [0, 0, width, height])
      .attr("font-family", fontFamily)
      .attr("text-anchor", "middle");

  const cloud = d3.cloud()
      .size([width, height])
      .words(data.map(d => Object.create(d)))
      .padding(padding)
      .rotate(rotate)
      .font(fontFamily)
      .fontSize(d => Math.sqrt(d.value) * fontScale)
      .on("word", ({size, x, y, rotate, text}) => {
        svg.append("text")
            .attr("font-size", size)
            .attr("transform", `translate(${x},${y}) rotate(${rotate})`)
            .text(text);
      });

  cloud.start();
  invalidation.then(() => cloud.stop());
  return svg.node();
}
