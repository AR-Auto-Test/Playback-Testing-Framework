package c.e.b.ff;

import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import androidx.databinding.ViewDataBinding;

/* compiled from: ItemHistoryBinding.java */
/* loaded from: classes2.dex */
public abstract class g extends ViewDataBinding {
    public final ImageView r;
    public final LinearLayout s;
    public c.e.b.hf.e t;

    public g(Object obj, View view, int i, ImageView imageView, LinearLayout linearLayout) {
        super(obj, view, i);
        this.r = imageView;
        this.s = linearLayout;
    }

    public abstract void m(c.e.b.hf.e eVar);
}