package c.b.a.a;

import android.net.Uri;
import android.util.Log;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.RecyclerView;
import b.d.b.d1.d1;
import b.d.b.d1.e1;
import b.d.b.d1.o;
import com.google.android.gms.internal.measurement.zzbl;
import com.google.android.gms.internal.measurement.zzh;
import com.google.android.gms.measurement.internal.zzfr;
import com.google.android.play.core.internal.zzag;
import com.google.android.play.core.tasks.zzi;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.Vertex;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ScaleController;
import com.google.ar.sceneform.ux.SimpleTransformableNode;
import com.google.protobuf.BinaryWriter;
import com.google.protobuf.DescriptorProtos;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.opencv.core.CvType;

/* compiled from: outline */
/* loaded from: classes.dex */
public class a {
    public static StringBuilder A(String str, String str2) {
        StringBuilder sb = new StringBuilder();
        sb.append(str);
        sb.append(str2);
        return sb;
    }

    public static StringBuilder B(String str, String str2, String str3) {
        StringBuilder sb = new StringBuilder();
        sb.append(str);
        sb.append(str2);
        sb.append(str3);
        return sb;
    }

    public static void C(float f2, float f3, float f4, Node node) {
        node.setLocalScale(new Vector3(f2, f3, f4));
    }

    public static void D(int i, int i2, StringBuilder sb, String str) {
        sb.append(i - i2);
        sb.append(str);
    }

    public static void E(zzfr zzfrVar, String str) {
        zzfrVar.zzay().zzd().zza(str);
    }

    public static void F(zzfr zzfrVar, String str) {
        zzfrVar.zzay().zzj().zza(str);
    }

    public static void G(zzfr zzfrVar, String str) {
        zzfrVar.zzay().zzk().zza(str);
    }

    public static o H(List list, o oVar, e1.b bVar, e1.a aVar) {
        list.add(oVar);
        return new o(bVar, aVar);
    }

    public static void I(Vector3 vector3, float f2, Node node, Light light) {
        node.setLocalScale(vector3.scaled(f2));
        node.setLight(light);
    }

    public static void J(ViewRenderable viewRenderable, ViewRenderable.VerticalAlignment verticalAlignment, boolean z, boolean z2) {
        viewRenderable.setVerticalAlignment(verticalAlignment);
        viewRenderable.setShadowCaster(z);
        viewRenderable.setShadowReceiver(z2);
    }

    public static void K(BinaryWriter binaryWriter, int i, int i2, int i3) {
        binaryWriter.writeVarint32(binaryWriter.getTotalBytesWritten() - i);
        binaryWriter.writeTag(i2, i3);
    }

    public static void L(String str, int i, String str2) {
        Log.d(str2, str + i);
    }

    public static void M(StringBuilder sb, Fragment fragment, String str) {
        sb.append(fragment);
        Log.d(str, sb.toString());
    }

    public static void N(Throwable th, StringBuilder sb, String str) {
        sb.append(th.getMessage());
        Log.e(str, sb.toString());
    }

    public static d1 O(List list, o oVar, ArrayList arrayList, d1 d1Var) {
        list.add(oVar);
        arrayList.add(d1Var);
        return new d1();
    }

    public static zzi P(zzag zzagVar, String str, Object[] objArr) {
        zzagVar.zzd(str, objArr);
        return new zzi();
    }

    public static Node Q(SimpleTransformableNode simpleTransformableNode, Vector3 vector3, SimpleTransformableNode simpleTransformableNode2) {
        simpleTransformableNode.setLocalPosition(vector3);
        Node node = new Node();
        node.setParent(simpleTransformableNode2);
        return node;
    }

    public static Vertex R(Vector3 vector3, Vector3 vector32, Vertex.UvCoordinate uvCoordinate) {
        return Vertex.builder().setPosition(vector3).setNormal(vector32).setUvCoordinate(uvCoordinate).build();
    }

    public static ViewRenderable.Builder S(float f2, float f3, ScaleController scaleController) {
        scaleController.setMaxScale(f2 * f3);
        return ViewRenderable.builder();
    }

    public static float a(float f2, float f3, float f4, float f5) {
        return ((f2 - f3) * f4) + f5;
    }

    public static int b(int i, int i2, int i3, int i4) {
        return ((i * i2) + i3) * i4;
    }

    public static ScaleController c(float f2, float f3, ScaleController scaleController, SimpleTransformableNode simpleTransformableNode) {
        scaleController.setMinScale(f2 * f3);
        return simpleTransformableNode.getScaleController();
    }

    public static Integer d(HashMap hashMap, Integer num, String str, int i, String str2) {
        hashMap.put(num, str);
        Integer valueOf = Integer.valueOf(i);
        hashMap.put(valueOf, str2);
        return valueOf;
    }

    public static Object e(int i) {
        return DescriptorProtos.getDescriptor().getMessageTypes().get(i);
    }

    public static Object f(zzbl zzblVar, int i, List list, int i2) {
        zzh.zzh(zzblVar.name(), i, list);
        return list.get(i2);
    }

    public static String g(int i, String str, int i2) {
        StringBuilder sb = new StringBuilder(i);
        sb.append(str);
        sb.append(i2);
        return sb.toString();
    }

    public static String h(int i, String str, int i2, String str2, int i3) {
        StringBuilder sb = new StringBuilder(i);
        sb.append(str);
        sb.append(i2);
        sb.append(str2);
        sb.append(i3);
        return sb.toString();
    }

    public static String i(RecyclerView recyclerView, StringBuilder sb) {
        sb.append(recyclerView.exceptionLabel());
        return sb.toString();
    }

    public static String j(String str, int i) {
        return str + i;
    }

    public static String k(String str, int i, String str2, int i2) {
        return str + i + str2 + i2;
    }

    public static String l(String str, long j) {
        return str + j;
    }

    public static int m(int i, int i2, int i3, int i4) {
        return ((i * i2) / i3) + i4;
    }

    public static String n(String str, Uri uri) {
        return str + uri;
    }

    public static String o(String str, Fragment fragment, String str2) {
        return str + fragment + str2;
    }

    public static String p(String str, Object obj) {
        return str + obj;
    }

    public static String q(String str, String str2) {
        return str + str2;
    }

    public static String r(String str, String str2, String str3) {
        return str + str2 + str3;
    }

    public static String s(StringBuilder sb, int i, String str) {
        sb.append(i);
        sb.append(str);
        return sb.toString();
    }

    public static String t(StringBuilder sb, int i, String str, int i2, String str2) {
        sb.append(i);
        sb.append(str);
        sb.append(CvType.channels(i2));
        sb.append(str2);
        return sb.toString();
    }

    public static String u(StringBuilder sb, Object obj, String str) {
        sb.append(obj);
        sb.append(str);
        return sb.toString();
    }

    public static String v(StringBuilder sb, String str, String str2) {
        sb.append(str);
        sb.append(str2);
        return sb.toString();
    }

    public static int w(ByteBuffer byteBuffer, ByteOrder byteOrder) {
        byteBuffer.order(byteOrder);
        return byteBuffer.getInt(byteBuffer.position());
    }

    public static StringBuilder x(String str) {
        StringBuilder sb = new StringBuilder();
        sb.append(str);
        return sb;
    }

    public static StringBuilder y(String str, int i, String str2) {
        StringBuilder sb = new StringBuilder();
        sb.append(str);
        sb.append(i);
        sb.append(str2);
        return sb;
    }

    public static StringBuilder z(String str, int i, String str2, int i2, String str3) {
        StringBuilder sb = new StringBuilder();
        sb.append(str);
        sb.append(i);
        sb.append(str2);
        sb.append(i2);
        sb.append(str3);
        return sb;
    }
}