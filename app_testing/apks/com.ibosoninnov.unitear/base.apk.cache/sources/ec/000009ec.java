package c.e.b.ef;

import android.app.Activity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;

/* compiled from: CategoryAdapter.java */
/* loaded from: classes2.dex */
public class c extends RecyclerView.g<a> {

    /* renamed from: a  reason: collision with root package name */
    public ArrayList<c.e.b.hf.d> f4712a;

    /* renamed from: b  reason: collision with root package name */
    public Activity f4713b;

    /* renamed from: c  reason: collision with root package name */
    public c.e.b.gf.a f4714c;

    /* compiled from: CategoryAdapter.java */
    /* loaded from: classes2.dex */
    public static class a extends RecyclerView.d0 {

        /* renamed from: a  reason: collision with root package name */
        public TextView f4715a;

        /* renamed from: b  reason: collision with root package name */
        public LinearLayout f4716b;

        public a(View view) {
            super(view);
            this.f4715a = (TextView) view.findViewById(R.id.cat_name);
            this.f4716b = (LinearLayout) view.findViewById(R.id.active_bar);
        }
    }

    public c(ArrayList<c.e.b.hf.d> arrayList, Activity activity) {
        this.f4712a = arrayList;
        this.f4713b = activity;
        this.f4714c = (c.e.b.gf.a) activity;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.g
    public int getItemCount() {
        return this.f4712a.size();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [androidx.recyclerview.widget.RecyclerView$d0, int] */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public void onBindViewHolder(a aVar, int i) {
        a aVar2 = aVar;
        aVar2.f4715a.setText(this.f4712a.get(i).name);
        if (this.f4712a.get(i).isSelected) {
            aVar2.f4715a.setTextColor(this.f4713b.getResources().getColor(R.color.blue3));
            aVar2.f4716b.setVisibility(0);
        } else {
            aVar2.f4715a.setTextColor(this.f4713b.getResources().getColor(R.color.grey_333333));
            aVar2.f4716b.setVisibility(4);
        }
        aVar2.itemView.setOnClickListener(new b(this, i));
    }

    /* JADX DEBUG: Return type fixed from 'androidx.recyclerview.widget.RecyclerView$d0' to match base method */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public a onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new a(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_category_name, viewGroup, false));
    }
}