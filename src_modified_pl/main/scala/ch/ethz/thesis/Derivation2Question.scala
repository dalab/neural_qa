package ch.ethz.thesis

import edu.stanford.nlp.sempre.SuperlativeFormula.Mode
import edu.stanford.nlp.sempre.{AggregateFormula, ArithmeticFormula, _}

/**
  * Created by tillhaug on 23.12.16.
  */
object Derivation2Question {
  def transform(derivation: Derivation, example: Example): String = {
    val formula = derivation.getFormula

    // Strip newlines and other control characters
    val result = translate(formula).map{c =>
      if(c >= ' ') {
        c
      }
      else {
        ' '
      }
    }

    result
  }


  def translate(formula: Formula): String = {
    formula match {
      case agg: AggregateFormula =>
        val aggName = agg.mode match {
          case AggregateFormula.Mode.count => "count"
          case AggregateFormula.Mode.sum => "total"
          case AggregateFormula.Mode.avg => "average"
          case AggregateFormula.Mode.min => "minimum"
          case AggregateFormula.Mode.max => "maximum"
        }
        aggName + " " + translate(agg.child)
      case join: JoinFormula =>
        val l = translate(join.relation)
        val r = translate(join.child)

        val isReverseL = join.relation.isInstanceOf[ReverseFormula]
        val isValueL = join.relation.isInstanceOf[ValueFormula[Value]]
        val isValueR = join.child.isInstanceOf[ValueFormula[Value]]

        if(isValueL && isValueR) {
          if(join.relation.toString == "fb:type.object.type" && join.child.toString == "fb:type.row") {
            "allrows"
          }
          else {
            s"$l is $r"
          }
        }
        else if(isReverseL && r.nonEmpty) {
          s"$l of $r"
        }
        else {
          s"$l $r"
        }
      case reverse: ReverseFormula =>
        translate(reverse.child)
      case nameValue: ValueFormula[Value] =>
        nameValue.value match {
          case nameValue: NameValue =>
            nameValue.id match {
              case "fb:cell.cell.number" => "numbernorm"
              case "fb:cell.cell.date" => "datenorm"
              case "!fb:row.row.next" => "rowafter"
              case "fb:row.row.next" => "rowbefore"
              case "fb:row.row.index" => "rownumber"
              case _ =>
                if(nameValue.description != null) {
                  nameValue.description
                }
                else if(nameValue.id != null) {
                  nameValue.id
                }
                else {
                  ""
                }
            }
          case numberValue: NumberValue =>
            val num = numberValue.value.toString
            if(num.endsWith(".0")) {
              numberValue.value.toInt.toString
            }
            else {
              num
            }
          case dateValue: DateValue =>
            Vector(dateValue.day, dateValue.month, dateValue.year).filter(_ != -1).mkString(" ")
        }
      case lambda: LambdaFormula =>
        translate(lambda.body)
      case _: VariableFormula =>
        ""
      case arith: ArithmeticFormula =>
        val l = translate(arith.child1)
        val r = translate(arith.child2)

        arith.mode match {
          case ArithmeticFormula.Mode.add => s"$l plus $r"
          case ArithmeticFormula.Mode.sub => s"$l minus $r"
          case ArithmeticFormula.Mode.mul => s"$l times $r"
          case ArithmeticFormula.Mode.div => s"$l divided $r"
        }
      case merge: MergeFormula =>
        val l = translate(merge.child1)
        val r = translate(merge.child2)
        val isAnd = merge.mode == MergeFormula.Mode.and

        if(isAnd) {
          s"$l intersection $r"
        }
        else {
          s"$l union $r"
        }

      case sup: SuperlativeFormula =>
        val overIndex = sup.relation.toString == "fb:row.row.index"
        val rel = translate(sup.relation)
        var head = translate(sup.head)

        if(head == "allrows") {
          head = ""
        }
        else {
          head = " where " + head
        }

        if(overIndex) {
          val supName = sup.mode match {
            case Mode.argmin => "firstrow"
            case Mode.argmax => "lastrow"
          }
          supName + head
        }
        else {
          val supName = sup.mode match {
            case Mode.argmin => "rowwithlowest"
            case Mode.argmax => "rowwithhighest"
          }
          supName + " " + rel + head
        }

    }
  }

}
