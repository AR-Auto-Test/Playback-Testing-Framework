package c.e.b.ff;

import android.util.SparseIntArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.databinding.ViewDataBinding;
import com.ibosoninnov.unitear.R;

/* compiled from: ItemMenuBindingImpl.java */
/* loaded from: classes2.dex */
public class j extends i {
    public static final SparseIntArray s;
    public final TextView t;
    public long u;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        s = sparseIntArray;
        sparseIntArray.put(R.id.imIcon, 2);
        sparseIntArray.put(R.id.view, 3);
    }

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public j(b.m.e eVar, View view) {
        super(eVar, view, 0, (ImageView) r0[2], (LinearLayout) r0[0], (View) r0[3]);
        Object[] h2 = ViewDataBinding.h(eVar, view, 4, s);
        this.u = -1L;
        this.r.setTag(null);
        TextView textView = (TextView) h2[1];
        this.t = textView;
        textView.setTag(null);
        view.setTag(R.id.dataBinding, this);
        synchronized (this) {
            this.u = 2L;
        }
        l();
    }

    @Override // androidx.databinding.ViewDataBinding
    public void c() {
        long j;
        synchronized (this) {
            j = this.u;
            this.u = 0L;
        }
        if ((j & 3) != 0) {
            b.j.b.d.Q(this.t, null);
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean e() {
        synchronized (this) {
            return this.u != 0;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean i(int i, Object obj, int i2) {
        return false;
    }
}