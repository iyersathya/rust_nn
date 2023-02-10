use std::cell::RefCell;
use std::fmt;
use std::iter::Sum;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
pub struct Value {
    pub data: Rc<RefCell<f64>>,
    pub grad: Rc<RefCell<f64>>,
    pub _prev: Vec<Arc<Value>>,
    pub _op: String,
    pub _label: RefCell<String>,
    pub _backward: Arc<Box<dyn Fn(&Value)>>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        *(*self.data).borrow() == *(*other.data).borrow()
            && *(*self.grad).borrow() == *(*other.grad).borrow()
            && self._prev == other._prev
            && self._op == other._op
            && self._label == other._label
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value {{ label: {}, data: {}, grad: {}, op: {}, _prev: {:#?}}}",
            &self._label.borrow(),
            (*self.data).borrow(),
            (*self.grad).borrow(),
            &self._op,
            self._prev,
        )
    }
}
impl Value {
    pub fn new(data: f64, child: Vec<Arc<Value>>, _op: String, label: String) -> Value {
        Value {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            _prev: child,
            _op,
            _label: RefCell::new(label),
            _backward: Arc::new(Box::new(|_: &Value| {})),
        }
    }
    pub fn vec(inv: &[f64]) -> Vec<Value> {
        inv.to_vec()
            .iter()
            .enumerate()
            .map(|(i, v)| Value::new(*v, vec![], "".to_string(), format!("x{}", i)))
            .collect()
    }
    pub fn get_data(&self) -> f64 {
        *(*self.data).borrow()
    }
    pub fn set_data(&self, d: f64) {
        let mut data = (*self.data).borrow_mut();
        *data = d;
    }
    pub fn set_grad(&self, d: f64) {
        let mut grad = (*self.grad).borrow_mut();
        *grad = d;
    }
    pub fn get_grad(&self) -> f64 {
        *(*self.grad).borrow()
    }
    pub fn get_label(&self) -> String {
        self._label.borrow().to_string()
    }
    fn set_label(&self, l: &str) {
        let mut label = self._label.borrow_mut();
        label.push_str(l);
    }
    pub fn zero_grad(&self) {
        *(*self.grad).borrow_mut() = 0.0;
    }

    pub fn backward(&self) {
        let mut topo = vec![];
        let mut visited = vec![];
        fn build_topo<'a>(v: &'a Value, topo: &mut Vec<&'a Value>, visited: &mut Vec<&'a Value>) {
            if !visited.contains(&v) {
                visited.push(v);
                for child in v._prev.iter() {
                    build_topo(child, topo, visited);
                }
                topo.push(v);
            }
        }
        build_topo(self, &mut topo, &mut visited);

        *(*self.grad).borrow_mut() = 1.0;
        for v in topo.iter().rev() {
            (v._backward)(v);
        }
    }
}

impl Sum for Value {
    fn sum<I>(iter: I) -> Value
    where
        I: Iterator<Item = Value>,
    {
        iter.fold(
            Value::new(0.0, vec![], "".to_string(), "".to_string()),
            |a, b| a + b,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add_values() {
        let x = Value::new(3.0, vec![], "".to_string(), "x".to_string());
        let y = Value::new(3.0, vec![], "".to_string(), "y".to_string());

        // let z = x.add(&y);
        let mut z = x.clone() + y.clone();
        z.set_label("z");
        assert_eq!(*(*z.data).borrow(), 6.0);
        z = z.clone() + Value::new(3.0, vec![], "".to_string(), "".to_string());
        assert_eq!(*(*z.data).borrow(), 9.0);
        println!(" z:{:#?}", z);
        let zz = z + 1.0;
        assert_eq!(*(*zz.data).borrow(), 10.0);
        let zz = 1.0 + zz;
        assert_eq!(*(*zz.data).borrow(), 11.0);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(*(*x.clone().grad).borrow(), 1.0);
        assert_eq!(*(*y.clone().grad).borrow(), 1.0);
    }

    #[test]
    fn test_sub_values() {
        let x = Value::new(3.0, vec![], "".to_string(), "x".to_string());
        let y = Value::new(5.0, vec![], "".to_string(), "y".to_string());
        // let z = x.add(&y);
        let z = x.clone() - y.clone();
        z.set_label("z");
        assert_eq!(*(*z.data).borrow(), -2.0);
        let z1 = z.clone() - Value::new(8.0, vec![], "".to_string(), "".to_string());
        z1.set_label("z1");
        assert_eq!(*(*z1.clone().data).borrow(), -10.0);
        println!(" z1:{:#?}", z1);
        let zz1 = z1 - 1.0;
        zz1.set_label("zz1");
        assert_eq!(*(*zz1.data).borrow(), -11.0);
        let zz2 = 1.0 - zz1;
        zz2.set_label("zz2");
        assert_eq!(*(*zz2.data).borrow(), 12.0);
        zz2.backward();
        println!(" z:{:#?}", zz2);
        assert_eq!(*(*x.clone().grad).borrow(), -1.0);
        assert_eq!(*(*y.clone().grad).borrow(), 1.0);
    }
    #[test]
    fn test_mul_values() {
        let x = Value::new(3.0, vec![], "".to_string(), "x".to_string());
        let y = Value::new(2.0, vec![], "".to_string(), "y".to_string());
        // let z = x.add(&y);
        let mut z = x.clone() * y.clone();
        assert_eq!(*(*z.data).borrow(), 6.0);
        z = z.clone() * Value::new(3.0, vec![], "".to_string(), "".to_string());
        assert_eq!(*(*z.data).borrow(), 18.0);
        println!(" z:{:#?}", z);
        let zz = z * 2.0;
        assert_eq!(*(*zz.data).borrow(), 36.0);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(*(*x.clone().grad).borrow(), 12.0);
        assert_eq!(*(*y.clone().grad).borrow(), 18.0);
    }

    #[test]
    fn test_div_values() {
        let x = Value::new(100.0, vec![], "".to_string(), "x".to_string());
        let y = Value::new(2.0, vec![], "".to_string(), "y".to_string());
        // let z = x.add(&y);
        let mut z = x.clone() / y.clone();
        assert_eq!(*(*z.data).borrow(), 50.0);
        z = z.clone() / Value::new(2.0, vec![], "".to_string(), "".to_string());
        assert_eq!(*(*z.data).borrow(), 25.0);
        println!(" z:{:#?}", z);
        let zz = z / 2.0;
        assert_eq!(*(*zz.data).borrow(), 12.5);
        let zz = 2.0 / zz;
        assert_eq!(*(*zz.data).borrow(), 0.16);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(*(*x.clone().grad).borrow(), -0.0016);
        assert_eq!(*(*y.clone().grad).borrow(), 0.08);
    }

    #[test]
    fn test_clone_values() {
        let a = Value::new(-2.0, vec![], "".to_string(), "a".to_string());
        let b = Value::new(3.0, vec![], "".to_string(), "b".to_string());
        let d = a.clone() * b.clone();
        d.set_label("d");

        let e = a.clone() + b.clone();
        e.set_label("e");

        let f = d.clone() * e.clone();
        f.set_label("f");
        println!("Before f:{:#?}", f);
        f.backward();
        println!("After f:{:#?}", f);
        assert_eq!(*(*a.clone().grad).borrow(), -3.0);
        assert_eq!(*(*b.clone().grad).borrow(), -8.0);
    }

    #[test]
    fn test_values() {
        let a = Value::new(2.0, vec![], "".to_string(), "a".to_string());
        let b = Value::new(-3.0, vec![], "".to_string(), "b".to_string());
        let c = Value::new(10.0, vec![], "".to_string(), "c".to_string());
        let e = a.clone() * b.clone();
        e.set_label("e");
        println!(" e {:#?}", *(*e.data).borrow());
        assert_eq!(*(*e.data).borrow(), -6.0);
        let d = e.clone() + c.clone();
        d.set_label("d");
        println!(" d {:#?}", *(*d.data).borrow());
        assert_eq!(*(*d.data).borrow(), 4.0);
        let f = Value::new(2.0, vec![], "".to_string(), "f".to_string());
        let L = d.clone() * f.clone();
        assert_eq!(*(*L.data).borrow(), 8.0);
        println!(" L {:#?}", *(*L.data).borrow());
        L.set_label("L");
        // println!(" {:#?}", L);
        L.backward();
        println!(" {:#?}", L);

        assert_eq!(*(*a.clone().grad).borrow(), -6.0);
        assert_eq!(*(*b.clone().grad).borrow(), 4.0);
        assert_eq!(*(*c.clone().grad).borrow(), 2.0);
        assert_eq!(*(*d.clone().grad).borrow(), 2.0);
        assert_eq!(*(*e.clone().grad).borrow(), 2.0);
        assert_eq!(*(*f.clone().grad).borrow(), 4.0);
    }

    #[test]
    fn test_activation_unit() {
        let x1 = Value::new(2.0, vec![], "".to_string(), "x1".to_string());
        let x2 = Value::new(0.0, vec![], "".to_string(), "x2".to_string());

        let w1 = Value::new(-3.0, vec![], "".to_string(), "w1".to_string());
        let w2 = Value::new(1.0, vec![], "".to_string(), "w2".to_string());

        let b = Value::new(6.8813, vec![], "".to_string(), "b".to_string());

        let x1w1 = x1.clone() * w1.clone();
        x1w1.set_label("x1w1");

        let x2w2 = x2.clone() * w2.clone();
        x2w2.set_label("x2w2");

        let x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.set_label("x1w1 + x2w2");

        let n = x1w1x2w2 + b;
        n.set_label("x1w1x2w2 + b");

        let o = n.clone().tanh();
        o.set_label("tanh");
        println!(" {:#?}", o);
        // o.grad.set(1.0);
        o.backward();
        println!(" {:#?}", o);
        assert_eq!(
            f64::trunc(*(*x1.clone().grad).borrow() * 100.0) / 100.0,
            -1.50
        );
        assert_eq!(
            f64::trunc(*(*w1.clone().grad).borrow() * 100.0) / 100.0,
            1.00
        );
        assert_eq!(
            f64::trunc(*(*x2.clone().grad).borrow() * 100.0) / 100.0,
            0.50
        );
        assert_eq!(
            f64::trunc(*(*w2.clone().grad).borrow() * 100.0) / 100.0,
            0.0
        );
    }

    #[test]
    fn test_activation_unit_exp() {
        let x1 = Value::new(2.0, vec![], "".to_string(), "x1".to_string());
        let x2 = Value::new(0.0, vec![], "".to_string(), "x2".to_string());

        let w1 = Value::new(-3.0, vec![], "".to_string(), "w1".to_string());
        let w2 = Value::new(1.0, vec![], "".to_string(), "w2".to_string());

        let b = Value::new(6.8813, vec![], "".to_string(), "b".to_string());

        let x1w1 = x1.clone() * w1.clone();
        x1w1.set_label("x1w1");

        let x2w2 = x2.clone() * w2.clone();
        x2w2.set_label("x2w2");

        let x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.set_label("x1w1 + x2w2");

        let n = x1w1x2w2 + b;
        n.set_label("x1w1x2w2 + b");

        let e = n.clone() * 2.0;
        let ee = e.exp();
        let o = (ee.clone() - 1.0) / (ee.clone() + 1.0);

        assert_eq!(
            f64::trunc(*(*o.clone().data).borrow() * 1000.0) / 1000.0,
            0.707
        );

        o.set_label("exp");
        println!(" {:#?}", o);
        // o.grad.set(1.0);
        o.backward();
        println!(" {:#?}", o);

        assert_eq!(
            f64::trunc(*(*x1.clone().grad).borrow() * 100.0) / 100.0,
            -1.50
        );
        assert_eq!(
            f64::trunc(*(*w1.clone().grad).borrow() * 100.0) / 100.0,
            1.00
        );
        assert_eq!(
            f64::trunc(*(*x2.clone().grad).borrow() * 100.0) / 100.0,
            0.50
        );
        assert_eq!(
            f64::trunc(*(*w2.clone().grad).borrow() * 100.0) / 100.0,
            0.0
        );
    }

    #[test]
    fn test_tanh_and_exp() {
        let x1 = Value::new(2.0, vec![], "".to_string(), "x1".to_string());
        let x2 = Value::new(3.0, vec![], "".to_string(), "x2".to_string());

        let n = x1.clone() * x2.clone();
        let o = n.clone().tanh();
        o.set_label("tanh");
        println!(" {:#?}", o);
        o.backward();
        println!(" {:#?}", o);
        println!("*****************************");
        let xx1 = Value::new(2.0, vec![], "".to_string(), "x1".to_string());
        let xx2 = Value::new(3.0, vec![], "".to_string(), "x2".to_string());
        let n = xx1.clone() * xx2.clone();
        n.set_label("n");
        let n1 = 2.0 * n.clone();
        n1.set_label("n1");
        let e1 = n1.exp();
        e1.set_label("e1");
        let n2 = 2.0 * n.clone();
        n2.set_label("n2");
        let e2 = n2.exp();
        e2.set_label("e2");
        let oo = (e1 - 1.0) / (e2 + 1.0);
        oo.set_label("oo");
        oo.set_label("exp");
        oo.backward();
        println!(" {:#?}", oo);
        assert_eq!(
            f64::trunc(*(*o.data).borrow() * 100.0) / 100.0,
            f64::trunc(*(*oo.data).borrow() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(*(*o.grad).borrow() * 100.0) / 100.0,
            f64::trunc(*(*oo.grad).borrow() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(*(*x1.grad).borrow() * 100.0) / 100.0,
            f64::trunc(*(*xx1.grad).borrow() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(*(*x2.grad).borrow() * 100.0) / 100.0,
            f64::trunc(*(*xx2.grad).borrow() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(*(*x1.grad).borrow() * 1000000.0) / 1000000.0,
            f64::trunc(*(*xx1.grad).borrow() * 1000000.0) / 1000000.0
        );
        assert_eq!(
            f64::trunc(*(*x2.grad).borrow() * 1000000.0) / 1000000.0,
            f64::trunc(*(*xx2.grad).borrow() * 1000000.0) / 1000000.0
        );
    }
}
