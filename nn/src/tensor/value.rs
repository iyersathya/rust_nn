use std::cell::RefCell;
use std::fmt;
use std::iter::Sum;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
pub struct Value {
    data: Rc<RefCell<f64>>,
    grad: Rc<RefCell<f64>>,
    pub _prev: Vec<Arc<Value>>,
    _op: Rc<String>,
    _label: RefCell<String>,
    // pub _backward: Arc<Box<dyn Fn(&Value)>>,
    pub _backward: Rc<fn(&Value)>,
}

impl Default for Value {
    fn default() -> Self {
        Value {
            data: Rc::new(RefCell::new(0.0)),
            grad: Rc::new(RefCell::new(0.0)),
            _prev: vec![],
            _op: Rc::new("".to_string()),
            _label: RefCell::new("".to_string()),
            _backward: Rc::new(|_: &Value| {}),
        }
    }
}

impl Value {
    pub fn new(data: f64, child: Vec<Arc<Value>>, _op: String, label: String) -> Value {
        Value {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            _prev: child,
            _op: Rc::new(_op),
            _label: RefCell::new(label),
            _backward: Rc::new(|_: &Value| {}),
        }
    }

    pub fn newd(data: f64, label: String) -> Value {
        Value {
            data: Rc::new(RefCell::new(data)),
            // _prev: child,
            _label: RefCell::new(label),
            ..Default::default()
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

    pub fn set_label(&self, l: &str) {
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
        assert_eq!(a.get_grad(), -3.0);
        assert_eq!(b.get_grad(), -8.0);
    }

    #[test]
    fn test_values() {
        let a = Value::new(2.0, vec![], "".to_string(), "a".to_string());
        let b = Value::new(-3.0, vec![], "".to_string(), "b".to_string());
        let c = Value::new(10.0, vec![], "".to_string(), "c".to_string());
        let e = a.clone() * b.clone();
        e.set_label("e");
        println!(" e {:#?}", *(*e.data).borrow());
        assert_eq!(e.get_data(), -6.0);
        let d = e.clone() + c.clone();
        d.set_label("d");
        println!(" d {:#?}", *(*d.data).borrow());
        assert_eq!(d.get_data(), 4.0);
        let f = Value::new(2.0, vec![], "".to_string(), "f".to_string());
        let L = d.clone() * f.clone();
        assert_eq!(L.get_data(), 8.0);
        println!(" L {:#?}", *(*L.data).borrow());
        L.set_label("L");
        // println!(" {:#?}", L);
        L.backward();
        println!(" {:#?}", L);

        assert_eq!(a.get_grad(), -6.0);
        assert_eq!(b.get_grad(), 4.0);
        assert_eq!(c.get_grad(), 2.0);
        assert_eq!(d.get_grad(), 2.0);
        assert_eq!(e.get_grad(), 2.0);
        assert_eq!(f.get_grad(), 4.0);
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
        assert_eq!(f64::trunc(x1.get_grad() * 100.0) / 100.0, -1.50);
        assert_eq!(f64::trunc(w1.get_grad() * 100.0) / 100.0, 1.00);
        assert_eq!(f64::trunc(x2.get_grad() * 100.0) / 100.0, 0.50);
        assert_eq!(f64::trunc(w2.get_grad() * 100.0) / 100.0, 0.0);
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

        assert_eq!(f64::trunc(o.get_data() * 1000.0) / 1000.0, 0.707);

        o.set_label("exp");
        println!(" {:#?}", o);
        // o.grad.set(1.0);
        o.backward();
        println!(" {:#?}", o);

        assert_eq!(f64::trunc(x1.get_grad() * 100.0) / 100.0, -1.50);
        assert_eq!(f64::trunc(w1.get_grad() * 100.0) / 100.0, 1.00);
        assert_eq!(f64::trunc(x2.get_grad() * 100.0) / 100.0, 0.50);
        assert_eq!(f64::trunc(w2.get_grad() * 100.0) / 100.0, 0.0);
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
            f64::trunc(o.get_data() * 100.0) / 100.0,
            f64::trunc(oo.get_data() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(o.get_grad() * 100.0) / 100.0,
            f64::trunc(oo.get_grad() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(x1.get_grad() * 100.0) / 100.0,
            f64::trunc(xx1.get_grad() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(x2.get_grad() * 100.0) / 100.0,
            f64::trunc(xx2.get_grad() * 100.0) / 100.0
        );
        assert_eq!(
            f64::trunc(x1.get_grad() * 1000000.0) / 1000000.0,
            f64::trunc(xx1.get_grad() * 1000000.0) / 1000000.0
        );
        assert_eq!(
            f64::trunc(x2.get_grad() * 1000000.0) / 1000000.0,
            f64::trunc(xx2.get_grad() * 1000000.0) / 1000000.0
        );
    }
}
