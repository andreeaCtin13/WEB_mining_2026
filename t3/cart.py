# from sample import sample
from math import log
from PIL import Image, ImageDraw
import util

# Nodul arborelui de decizie.
# Poate fi:
# - nod de decizie: are col, value, falseSubtree, trueSubtree
# - nod frunză: are results
class DecisionNode:
  def __init__(self, col=-1, value=None, results=None, falseSubtree=None, trueSubtree=None):
    self.col = col
    self.value = value
    self.results = results
    self.falseSubtree = falseSubtree
    self.trueSubtree = trueSubtree


# Numără câte exemple există pentru fiecare clasă.
# Ultima coloană din fiecare rând este clasa/rezultatul.
def uniquecounts(rows):
  results = {}

  for row in rows:
    outcome = row[-1]
    results.setdefault(outcome, 0)
    results[outcome] += 1

  return results


# Calculează entropia.
# Entropia măsoară cât de amestecate sunt clasele.
# Entropie mare = date amestecate.
# Entropie mică = date mai clare.
def entropy(rows):
  log2 = lambda x: log(x) / log(2)

  results = uniquecounts(rows)
  h = 0.0

  for result, count in results.items():
    p = float(count) / len(rows)
    h -= p * log2(p)

  return h


# Împarte datasetul în două:
# - trueSet: rândurile care respectă condiția
# - falseSet: rândurile care nu respectă condiția
def divideSet(rows, column, value):
  if isinstance(value, int) or isinstance(value, float):
    splitFunction = lambda row: row[column] >= value
  else:
    splitFunction = lambda row: row[column] == value

  trueSet = [row for row in rows if splitFunction(row)]
  falseSet = [row for row in rows if not splitFunction(row)]

  return falseSet, trueSet


# Construiește arborele de decizie.
def buildTree(rows, scoref=entropy):
  # Dacă nu avem date, întoarcem un nod gol.
  if len(rows) == 0:
    return DecisionNode()

  # Entropia inițială a datasetului.
  currentScore = scoref(rows)

  bestGain = 0.0
  bestCriterion = None
  bestSets = None

  # Ultima coloană este clasa, deci nu o folosim ca feature.
  columnCount = len(rows[0]) - 1

  # Parcurgem fiecare coloană/feature.
  for column in range(0, columnCount):

    # Strângem toate valorile unice din coloana curentă.
    columnValues = {}

    for row in rows:
      columnValues[row[column]] = 1

    # Testăm fiecare valoare ca posibilă regulă de split.
    for value in columnValues.keys():
      falseSet, trueSet = divideSet(rows, column, value)

      # Dacă una dintre părți este goală, split-ul nu e util.
      if len(falseSet) == 0 or len(trueSet) == 0:
        continue

      # Proporția exemplelor din falseSet.
      p = float(len(falseSet)) / len(rows)

      # Information Gain:
      # câtă entropie reducem dacă împărțim datele după această regulă.
      gain = currentScore - p * scoref(falseSet) - (1 - p) * scoref(trueSet)

      # Păstrăm cel mai bun split găsit până acum.
      if gain > bestGain:
        bestGain = gain
        bestCriterion = (column, value)
        bestSets = (falseSet, trueSet)

  # Dacă am găsit un split bun, construim recursiv subarborii.
  if bestGain > 0.0:
    falseBranch = buildTree(bestSets[0], scoref)
    trueBranch = buildTree(bestSets[1], scoref)

    return DecisionNode(
      col=bestCriterion[0],
      value=bestCriterion[1],
      trueSubtree=trueBranch,
      falseSubtree=falseBranch
    )

  # Dacă nu mai avem split bun, nodul devine frunză.
  else:
    return DecisionNode(results=uniquecounts(rows))


# Clasifică un element nou folosind arborele construit.
def classify(item, node):
  # Dacă nodul este frunză, întoarcem rezultatul.
  if node.results is not None:
    return node.results

  # Luăm valoarea itemului de pe coloana folosită în nod.
  value = item[node.col]

  # Pentru valori numerice folosim >=.
  if isinstance(value, int) or isinstance(value, float):
    print("decision column", node.col, ">= value", node.value)

    if value >= node.value:
      branch = node.trueSubtree
    else:
      branch = node.falseSubtree

  # Pentru valori text/categorice folosim ==.
  else:
    print("decision column", node.col, "== value", node.value)

    if value == node.value:
      branch = node.trueSubtree
    else:
      branch = node.falseSubtree

  # Continuăm recursiv până ajungem la o frunză.
  return classify(item, branch)


# Afișează arborele în consolă.
def printTree(node, indent=""):
  # Dacă e frunză, afișăm clasele și numărul de exemple.
  if node.results is not None:
    print(node.results)

  # Dacă e nod de decizie, afișăm condiția.
  else:
    print(node.col, ":", node.value, "?")

    print(indent, "True -> ", end="")
    printTree(node.trueSubtree, indent=indent + "   ")

    print(indent, "False -> ", end="")
    printTree(node.falseSubtree, indent=indent + "   ")


# Calculează lățimea arborelui pentru desenare.
def getWidth(node):
  if node.trueSubtree is None and node.falseSubtree is None:
    return 1

  return getWidth(node.trueSubtree) + getWidth(node.falseSubtree)


# Calculează înălțimea arborelui pentru desenare.
def getHeight(node):
  if node.trueSubtree is None and node.falseSubtree is None:
    return 0

  return 1 + max(getHeight(node.trueSubtree), getHeight(node.falseSubtree))


# Desenează arborele într-un fișier imagine.
def drawTree(tree, file="tree.jpg"):
  w = getWidth(tree) * 100
  h = getHeight(tree) * 100 + 120

  img = Image.new("RGB", (w, h), (255, 255, 255))
  draw = ImageDraw.Draw(img)

  drawNode(draw, tree, w / 2, 20)

  img.save(file, "JPEG")


# Desenează fiecare nod al arborelui.
def drawNode(draw, node, x, y):
  # Dacă e frunză, scriem rezultatele.
  if node.results is not None:
    results = [
      "{outcome}:{count}".format(outcome=k, count=v)
      for k, v in node.results.items()
    ]

    text = ", ".join(results)
    draw.text((x - 20, y), text, (0, 0, 0))

  # Dacă e nod de decizie, desenăm ramurile.
  else:
    falseWidth = getWidth(node.falseSubtree) * 100
    trueWidth = getWidth(node.trueSubtree) * 100

    left = x - (trueWidth + falseWidth) / 2
    right = x + (trueWidth + falseWidth) / 2

    draw.text((x - 20, y - 10), str(node.col) + ":" + str(node.value), (0, 0, 0))

    # Ramura falsă
    draw.line((x, y, left + falseWidth / 2, y + 100), fill=(255, 0, 0))

    # Ramura adevărată
    draw.line((x, y, right - trueWidth / 2, y + 100), fill=(0, 255, 0))

    drawNode(draw, node.falseSubtree, left + falseWidth / 2, y + 100)
    drawNode(draw, node.trueSubtree, right - trueWidth / 2, y + 100)


# https://archive.ics.uci.edu/ml/machine-learning-databases/car/
def main():
  # data = sample
  # Dacă vrei să testezi pe datasetul mic din sample.py, folosești linia de mai sus.

  data = util.loadCarData()
  # Încărcăm datasetul Car din util.py.

  # tree = buildTree(sample)
  # Dacă folosești sample, construiești arborele pe sample.

  tree = buildTree(data)
  # Construim arborele de decizie pe datele încărcate.

  printTree(tree)
  # Afișăm arborele în consolă.

  drawTree(tree)
  # Salvăm arborele desenat în fișierul tree.jpg.


if __name__ == "__main__":
  main()