package c.e.b.ff;

import android.util.SparseIntArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import androidx.databinding.ViewDataBinding;
import com.ibosoninnov.unitear.R;

/* compiled from: ItemArobjectsBindingImpl.java */
/* loaded from: classes2.dex */
public class f extends e {
    public static final SparseIntArray s;
    public long t;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        s = sparseIntArray;
        sparseIntArray.put(R.id.imIcon, 1);
    }

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public f(b.m.e eVar, View view) {
        super(eVar, view, 0, (ImageView) r0[1], (LinearLayout) r0[0]);
        Object[] h2 = ViewDataBinding.h(eVar, view, 2, s);
        this.t = -1L;
        this.r.setTag(null);
        view.setTag(R.id.dataBinding, this);
        synchronized (this) {
            this.t = 2L;
        }
        l();
    }

    @Override // androidx.databinding.ViewDataBinding
    public void c() {
        synchronized (this) {
            this.t = 0L;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean e() {
        synchronized (this) {
            return this.t != 0;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean i(int i, Object obj, int i2) {
        return false;
    }
}