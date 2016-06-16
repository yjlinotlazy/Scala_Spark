/**
  * Strangely I can't find certain row matrix multiplication implementations so I wrote some
*/

import breeze.linalg
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

object MatrixCalc {
  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    val prod = X.rows.map {case (row:Vector)=>
      val v1 = toBreezeVector(row)
      val prodd:DenseMatrix[Double] = (v1.toDenseVector.toDenseMatrix * d)
      prodd(0,::).t
    }.map(fromBreeze)
    new RowMatrix(prod)
  }
  
  /** row matrix multiplication of W.t and V */
  /** Naive implementation based on matrix mult. properties */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    // first need to align rows of W and V
    val WV = W.rows.zip(V.rows)
    
    // iterate over rows of W
    val prod1:RDD[Array[Vector]] = WV.map {case (a:Vector,b:Vector)=>
        val matrix1 = a.toArray.map {case (aa:Double)=>
          // iterate over b vector
          val mat = toBreezeVector(b) * aa
          mat
        }.map(fromBreeze)
      matrix1
    }
    val prod2 = prod1.reduce((x,y)=>addArrayVectors(x,y))
    val num_rows = W.numCols().toInt
    val num_cols = V.numCols().toInt
    var count = 0
    val bm= BDM.zeros[Double](num_rows,num_cols)
    prod2.foreach { case (v:Vector)=>
      bm(count,::) := toBreezeVector(v).t
      count += 1
    }
    bm
  }
}
  
