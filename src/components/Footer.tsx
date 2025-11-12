export function Footer() {
  return (
    <footer className="fixed bottom-0 left-0 right-0 w-full bg-[#005496] py-1">
      <div className="w-full px-5 flex items-center">
        {/* มุมซ้าย: รูปภาพชิดซ้ายสุด */}
        <div className="flex items-center">
          <img
            src="/autoliv_logo.png"
            alt="Logo"
            className="h-8 w-auto object-contain"
          />
        </div>
      </div>
    </footer>
  );
}