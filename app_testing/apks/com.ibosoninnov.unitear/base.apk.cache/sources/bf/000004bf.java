package b.k.a;

import android.content.Context;
import android.database.ContentObserver;
import android.database.Cursor;
import android.database.DataSetObserver;
import android.os.Handler;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Filter;
import android.widget.Filterable;
import b.k.a.b;

/* compiled from: CursorAdapter.java */
/* loaded from: classes.dex */
public abstract class a extends BaseAdapter implements Filterable, b.a {

    /* renamed from: b  reason: collision with root package name */
    public boolean f2304b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f2305c;

    /* renamed from: d  reason: collision with root package name */
    public Cursor f2306d;

    /* renamed from: e  reason: collision with root package name */
    public Context f2307e;

    /* renamed from: f  reason: collision with root package name */
    public int f2308f;

    /* renamed from: g  reason: collision with root package name */
    public C0040a f2309g;

    /* renamed from: h  reason: collision with root package name */
    public DataSetObserver f2310h;
    public b.k.a.b i;

    /* compiled from: CursorAdapter.java */
    /* renamed from: b.k.a.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0040a extends ContentObserver {
        public C0040a() {
            super(new Handler());
        }

        @Override // android.database.ContentObserver
        public boolean deliverSelfNotifications() {
            return true;
        }

        @Override // android.database.ContentObserver
        public void onChange(boolean z) {
            Cursor cursor;
            a aVar = a.this;
            if (!aVar.f2305c || (cursor = aVar.f2306d) == null || cursor.isClosed()) {
                return;
            }
            aVar.f2304b = aVar.f2306d.requery();
        }
    }

    /* compiled from: CursorAdapter.java */
    /* loaded from: classes.dex */
    public class b extends DataSetObserver {
        public b() {
        }

        @Override // android.database.DataSetObserver
        public void onChanged() {
            a aVar = a.this;
            aVar.f2304b = true;
            aVar.notifyDataSetChanged();
        }

        @Override // android.database.DataSetObserver
        public void onInvalidated() {
            a aVar = a.this;
            aVar.f2304b = false;
            aVar.notifyDataSetInvalidated();
        }
    }

    public a(Context context, Cursor cursor, boolean z) {
        boolean z2 = z ? true : true;
        if (z2 & true) {
            z2 |= true;
            this.f2305c = true;
        } else {
            this.f2305c = false;
        }
        boolean z3 = cursor != null;
        this.f2306d = cursor;
        this.f2304b = z3;
        this.f2307e = context;
        this.f2308f = z3 ? cursor.getColumnIndexOrThrow("_id") : -1;
        if (z2 & true) {
            this.f2309g = new C0040a();
            this.f2310h = new b();
        } else {
            this.f2309g = null;
            this.f2310h = null;
        }
        if (z3) {
            C0040a c0040a = this.f2309g;
            if (c0040a != null) {
                cursor.registerContentObserver(c0040a);
            }
            DataSetObserver dataSetObserver = this.f2310h;
            if (dataSetObserver != null) {
                cursor.registerDataSetObserver(dataSetObserver);
            }
        }
    }

    public abstract void a(View view, Context context, Cursor cursor);

    public void b(Cursor cursor) {
        Cursor cursor2 = this.f2306d;
        if (cursor == cursor2) {
            cursor2 = null;
        } else {
            if (cursor2 != null) {
                C0040a c0040a = this.f2309g;
                if (c0040a != null) {
                    cursor2.unregisterContentObserver(c0040a);
                }
                DataSetObserver dataSetObserver = this.f2310h;
                if (dataSetObserver != null) {
                    cursor2.unregisterDataSetObserver(dataSetObserver);
                }
            }
            this.f2306d = cursor;
            if (cursor != null) {
                C0040a c0040a2 = this.f2309g;
                if (c0040a2 != null) {
                    cursor.registerContentObserver(c0040a2);
                }
                DataSetObserver dataSetObserver2 = this.f2310h;
                if (dataSetObserver2 != null) {
                    cursor.registerDataSetObserver(dataSetObserver2);
                }
                this.f2308f = cursor.getColumnIndexOrThrow("_id");
                this.f2304b = true;
                notifyDataSetChanged();
            } else {
                this.f2308f = -1;
                this.f2304b = false;
                notifyDataSetInvalidated();
            }
        }
        if (cursor2 != null) {
            cursor2.close();
        }
    }

    public abstract CharSequence c(Cursor cursor);

    public abstract View d(Context context, Cursor cursor, ViewGroup viewGroup);

    @Override // android.widget.Adapter
    public int getCount() {
        Cursor cursor;
        if (!this.f2304b || (cursor = this.f2306d) == null) {
            return 0;
        }
        return cursor.getCount();
    }

    @Override // android.widget.BaseAdapter, android.widget.SpinnerAdapter
    public View getDropDownView(int i, View view, ViewGroup viewGroup) {
        if (this.f2304b) {
            this.f2306d.moveToPosition(i);
            if (view == null) {
                c cVar = (c) this;
                view = cVar.l.inflate(cVar.k, viewGroup, false);
            }
            a(view, this.f2307e, this.f2306d);
            return view;
        }
        return null;
    }

    @Override // android.widget.Filterable
    public Filter getFilter() {
        if (this.i == null) {
            this.i = new b.k.a.b(this);
        }
        return this.i;
    }

    @Override // android.widget.Adapter
    public Object getItem(int i) {
        Cursor cursor;
        if (!this.f2304b || (cursor = this.f2306d) == null) {
            return null;
        }
        cursor.moveToPosition(i);
        return this.f2306d;
    }

    @Override // android.widget.Adapter
    public long getItemId(int i) {
        Cursor cursor;
        if (this.f2304b && (cursor = this.f2306d) != null && cursor.moveToPosition(i)) {
            return this.f2306d.getLong(this.f2308f);
        }
        return 0L;
    }

    @Override // android.widget.Adapter
    public View getView(int i, View view, ViewGroup viewGroup) {
        if (this.f2304b) {
            if (this.f2306d.moveToPosition(i)) {
                if (view == null) {
                    view = d(this.f2307e, this.f2306d, viewGroup);
                }
                a(view, this.f2307e, this.f2306d);
                return view;
            }
            throw new IllegalStateException(c.b.a.a.a.j("couldn't move cursor to position ", i));
        }
        throw new IllegalStateException("this should only be called when the cursor is valid");
    }
}